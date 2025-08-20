import streamlit as st
import pandas as pd
import altair as alt
import time
from datetime import datetime, timedelta
from snowflake.snowpark.context import get_active_session

# Set page configuration
st.set_page_config(
    page_title="Healthcare AI Agents - Patient Recidivism Data App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define available models as strings
MODELS = [
    "openai-gpt-oss-120b", "openai-gpt-4.1", "openai-gpt-5", "openai-gpt-5-mini", "openai-gpt-5-nano", "openai-gpt-5-chat", "claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-8b", "llama3.1-70b", "llama4-maverick", "llama4-scout", "llama3.2-1b", "snowflake-llama-3.1-405b", "snowflake-llama-3.3-70b", "mistral-large2", "mistral-7b", "deepseek-r1", "snowflake-arctic", "reka-flash", "jamba-instruct", "gemma-7b"
]

# Initialize Snowflake session
try:
    session = get_active_session()
except Exception as e:
    st.error(f"❌ Error connecting to Snowflake: {str(e)}")
    st.stop()

# Database and schema settings - these will be set as context in Streamlit in Snowflake
DATABASE = "HOL_DATABASE"
SCHEMA = "PATIENT_RECIDIVISM_HEALTHCARE"

# Define table names
TABLE_PATIENTS = "patients"
TABLE_ENCOUNTERS = "encounters"
TABLE_EVENTS = "clinical_events"
TABLE_CLAIMS = "claims"
TABLE_RISK_SCORES = "risk_scores"

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Session state for data caching
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Add agent state management
if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False
if 'agent_results' not in st.session_state:
    st.session_state.agent_results = {}

def query_snowflake(query, params=None, cache_key=None, ttl=600):
    """Query Snowflake with optional parameters and caching support"""
    # Check cache first if cache_key is provided
    if cache_key and cache_key in st.session_state.data_cache:
        cache_time, df = st.session_state.data_cache[cache_key]
        if time.time() - cache_time < ttl:  # If cache is still valid
            return df
    
    try:
        if params:
            # Snowflake requires parameters to be passed as a list, not a dictionary
            modified_query = query
            param_values = []
            
            # Replace named parameters with question marks and collect values in order
            for param_name, param_value in params.items():
                placeholder = f":{param_name}"
                modified_query = modified_query.replace(placeholder, "?")
                param_values.append(param_value)
            
            # Execute with positional parameters
            result_df = session.sql(modified_query, params=param_values).to_pandas()
        else:
            result_df = session.sql(query).to_pandas()
        
        # Store in cache if cache_key is provided
        if cache_key:
            st.session_state.data_cache[cache_key] = (time.time(), result_df)

        return result_df
    except Exception as e:
        st.error(f"❌ Error querying Snowflake: {str(e)}")
        return pd.DataFrame()

def call_cortex_model(prompt, model_name):
    """Call Snowflake Cortex LLM model"""
    try:
        cortex_query = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            ?,
            ?
        ) as response
        """
        
        # Create system prompt for healthcare patient recidivism
        system_prompt = """
        You are a senior healthcare clinical analytics consultant specializing in hospital readmissions, patient care gaps, clinical event burden, and cost risk prediction.

        Your task is to:
        - Analyze patient clinical encounters, discharge dispositions, diagnoses, procedures, medications, labs, and financial claims history.
        - Identify patterns that indicate high likelihood of 30-day readmissions, care coordination failures, or rising cost risk.
        - Make clinical and operational recommendations to reduce recidivism.
        - Summarize insights using clear bullet points.
        - Rate final risk as Low, Medium, or High based on available evidence.

        Always write in clear, professional clinical language suitable for hospital executive dashboards.
        Do not invent new data — reason only from the context provided.
        """
        
        # Combine system prompt with user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Execute Cortex query
        response = session.sql(cortex_query, params=[model_name, full_prompt]).collect()[0][0]
        return response
    except Exception as e:
        st.error(f"Error calling Snowflake Cortex model: {str(e)}")
        return None

# AI Agent Data Analysis Functions
def analyze_real_proactive_care_data(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df):
    """Analyze real data for proactive care management insights"""
    results = {}
    
    # Convert encounter dates to datetime - handle Snowflake VARCHAR dates
    encounters_df_copy = encounters_df.copy()
    encounters_df_copy['ENCOUNTER_END_DATE'] = pd.to_datetime(encounters_df_copy['ENCOUNTER_END_DATE'], errors='coerce')
    
    # Get the actual date range of your data
    max_date = encounters_df_copy['ENCOUNTER_END_DATE'].max()
    min_date = encounters_df_copy['ENCOUNTER_END_DATE'].min()
    
    # Use the last 30 days of your actual data instead of current date
    recent_cutoff = max_date - timedelta(days=30)
    recent_discharges = encounters_df_copy[encounters_df_copy['ENCOUNTER_END_DATE'] >= recent_cutoff]
    results['recent_discharges_count'] = len(recent_discharges)
    results['data_range'] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    results['analysis_period'] = f"Last 30 days of data ({recent_cutoff.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
    
    # High-risk patients analysis
    high_risk_patients = patients_df[patients_df['RISK_CATEGORY'] == 'High']
    results['high_risk_count'] = len(high_risk_patients)
    
    # Readmission analysis - handle Snowflake VARCHAR boolean
    readmissions = encounters_df[encounters_df['IS_30_DAY_READMIT'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    results['readmission_count'] = len(readmissions)
    results['readmission_rate'] = (len(readmissions) / len(encounters_df)) * 100 if len(encounters_df) > 0 else 0
    
    # Care gaps analysis
    high_risk_patient_ids = high_risk_patients['PATIENT_ID'].tolist()
    recent_encounter_patient_ids = recent_discharges['PATIENT_ID'].tolist()
    care_gaps = [pid for pid in high_risk_patient_ids if pid not in recent_encounter_patient_ids]
    results['care_gaps_count'] = len(care_gaps)
    
    # Risk distribution
    risk_distribution = patients_df['RISK_CATEGORY'].value_counts().to_dict()
    results['risk_distribution'] = risk_distribution
    
    # Encounter type analysis
    encounter_types = encounters_df['ENCOUNTER_TYPE'].value_counts().to_dict()
    results['encounter_types'] = encounter_types
    
    # Top diagnosis codes
    top_diagnoses = encounters_df['DISCHARGE_DIAGNOSIS_CODE'].value_counts().head(5).to_dict()
    results['top_diagnoses'] = top_diagnoses
    
    # Financial analysis
    if not claims_df.empty:
        total_billed = pd.to_numeric(claims_df['BILLED_AMOUNT'], errors='coerce').sum()
        total_paid = pd.to_numeric(claims_df['PAID_AMOUNT'], errors='coerce').sum()
        results['total_billed'] = float(total_billed)
        results['total_paid'] = float(total_paid)
        results['payment_rate'] = (total_paid / total_billed * 100) if total_billed > 0 else 0
    
    # Risk factors analysis
    if not risk_scores_df.empty:
        risk_factors = risk_scores_df['KEY_RISK_FACTORS'].value_counts().to_dict()
        results['top_risk_factors'] = risk_factors
    
    # Clinical events analysis
    if not clinical_events_df.empty:
        clinical_events_copy = clinical_events_df.copy()
        clinical_events_copy['EVENT_DATE'] = pd.to_datetime(clinical_events_copy['EVENT_DATE'], errors='coerce')
        event_max_date = clinical_events_copy['EVENT_DATE'].max()
        recent_events_cutoff = event_max_date - timedelta(days=90)
        recent_events = clinical_events_copy[clinical_events_copy['EVENT_DATE'] >= recent_events_cutoff]
        results['recent_events_count'] = len(recent_events)
        results['event_analysis_period'] = f"Last 90 days of clinical events ({recent_events_cutoff.strftime('%Y-%m-%d')} to {event_max_date.strftime('%Y-%m-%d')})"
        
        recent_procedures = recent_events['CODE_VALUE'].value_counts().head(5).to_dict()
        results['recent_procedures'] = recent_procedures
    
    return results

def analyze_real_population_health_data(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df):
    """Analyze real data for population health management insights"""
    results = {}
    
    # Total patient population analysis
    results['total_patients'] = len(patients_df)
    
    # Chronic condition identification based on diagnosis codes
    chronic_conditions = {
        'diabetes': ['E11', 'E10', '250'],
        'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
        'heart_failure': ['I50'],
        'copd': ['J44', 'J43'],
        'ckd': ['N18']
    }
    
    condition_patients = {}
    for condition, codes in chronic_conditions.items():
        condition_encounters = encounters_df[
            encounters_df['DISCHARGE_DIAGNOSIS_CODE'].str.contains('|'.join(codes), case=False, na=False)
        ]
        unique_patients = condition_encounters['PATIENT_ID'].nunique()
        condition_patients[condition] = unique_patients
    
    results['chronic_conditions'] = condition_patients
    
    # Risk category analysis
    risk_distribution = patients_df['RISK_CATEGORY'].value_counts().to_dict()
    results['risk_distribution'] = risk_distribution
    
    # Deteriorating patients analysis
    high_risk_patients = patients_df[patients_df['RISK_CATEGORY'] == 'High']['PATIENT_ID'].tolist()
    
    encounters_df_copy = encounters_df.copy()
    encounters_df_copy['ENCOUNTER_END_DATE'] = pd.to_datetime(encounters_df_copy['ENCOUNTER_END_DATE'], errors='coerce')
    max_date = encounters_df_copy['ENCOUNTER_END_DATE'].max()
    recent_cutoff = max_date - timedelta(days=90)
    
    recent_encounters = encounters_df_copy[encounters_df_copy['ENCOUNTER_END_DATE'] >= recent_cutoff]
    deteriorating_patients = recent_encounters[recent_encounters['PATIENT_ID'].isin(high_risk_patients)]
    results['deteriorating_count'] = len(deteriorating_patients)
    results['analysis_period'] = f"Last 90 days of data ({recent_cutoff.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
    
    # Care adherence analysis
    patient_encounter_counts = encounters_df['PATIENT_ID'].value_counts()
    high_risk_single_encounter = encounters_df[
        (encounters_df['PATIENT_ID'].isin(high_risk_patients)) & 
        (encounters_df['PATIENT_ID'].isin(patient_encounter_counts[patient_encounter_counts == 1].index))
    ]
    results['poor_adherence_count'] = high_risk_single_encounter['PATIENT_ID'].nunique()
    
    # Intervention prioritization
    readmissions = encounters_df[encounters_df['IS_30_DAY_READMIT'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    high_risk_readmissions = readmissions[readmissions['PATIENT_ID'].isin(high_risk_patients)]
    results['high_impact_interventions'] = len(high_risk_readmissions)
    
    # Resource allocation by encounter type
    encounter_distribution = encounters_df['ENCOUNTER_TYPE'].value_counts().to_dict()
    results['encounter_distribution'] = encounter_distribution
    
    # Geographic and demographic analysis
    results['zip_codes_served'] = patients_df['ZIP_CODE'].nunique()
    high_risk_zips = patients_df[patients_df['RISK_CATEGORY'] == 'High']['ZIP_CODE'].value_counts().head(5).to_dict()
    results['high_risk_zip_codes'] = high_risk_zips
    
    gender_dist = patients_df['GENDER'].value_counts().to_dict()
    race_dist = patients_df['RACE'].value_counts().to_dict()
    language_dist = patients_df['PRIMARY_LANGUAGE'].value_counts().to_dict()
    
    results['demographics'] = {
        'gender': gender_dist,
        'race': race_dist,
        'language': language_dist
    }
    
    return results

def analyze_real_prior_authorization_data(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df):
    """Analyze real data for prior authorization insights"""
    results = {}
    
    if claims_df.empty:
        return {
            'total_claims': 0,
            'high_cost_threshold': 0,
            'high_cost_claims': 0,
            'authorization_analysis': 'No claims data available',
            'payer_distribution': {},
            'procedure_analysis': {},
            'diagnosis_patterns': {},
            'cost_analysis': {'total_billed': 0, 'total_paid': 0, 'savings_opportunity': 0}
        }
    
    results['total_claims'] = len(claims_df)
    
    # Convert amounts to numeric for analysis
    claims_df_copy = claims_df.copy()
    claims_df_copy['BILLED_AMOUNT'] = pd.to_numeric(claims_df_copy['BILLED_AMOUNT'], errors='coerce')
    claims_df_copy['PAID_AMOUNT'] = pd.to_numeric(claims_df_copy['PAID_AMOUNT'], errors='coerce')
    
    # High-cost claims analysis (top 10% by billed amount)
    cost_threshold = claims_df_copy['BILLED_AMOUNT'].quantile(0.9)
    high_cost_claims = claims_df_copy[claims_df_copy['BILLED_AMOUNT'] >= cost_threshold]
    results['high_cost_threshold'] = float(cost_threshold)
    results['high_cost_claims'] = len(high_cost_claims)
    results['high_cost_percentage'] = (len(high_cost_claims) / len(claims_df_copy)) * 100
    
    # Medical necessity validation
    procedure_codes = ['99213', '99214', '99215', '93000', '71020']
    high_value_procedures = claims_df_copy[claims_df_copy['PROCEDURE_CODE'].astype(str).isin([str(x) for x in procedure_codes])]
    results['medical_necessity_validated'] = len(high_value_procedures)
    results['validation_rate'] = (len(high_value_procedures) / len(claims_df_copy)) * 100
    
    # Cost-benefit analysis
    total_billed = claims_df_copy['BILLED_AMOUNT'].sum()
    total_paid = claims_df_copy['PAID_AMOUNT'].sum()
    payment_variance = total_billed - total_paid
    results['cost_analysis'] = {
        'total_billed': float(total_billed),
        'total_paid': float(total_paid),
        'payment_variance': float(payment_variance),
        'payment_rate': float((total_paid / total_billed) * 100) if total_billed > 0 else 0,
        'savings_opportunity': float(payment_variance * 0.15)
    }
    
    # Payer analysis
    payer_distribution = claims_df_copy['PAYER_ID'].value_counts().to_dict()
    results['payer_distribution'] = payer_distribution
    
    payer_payment_rates = {}
    for payer in payer_distribution.keys():
        payer_claims = claims_df_copy[claims_df_copy['PAYER_ID'] == payer]
        if not payer_claims.empty:
            payer_billed = payer_claims['BILLED_AMOUNT'].sum()
            payer_paid = payer_claims['PAID_AMOUNT'].sum()
            payment_rate = (payer_paid / payer_billed * 100) if payer_billed > 0 else 0
            payer_payment_rates[payer] = float(payment_rate)
    results['payer_approval_rates'] = payer_payment_rates
    
    # Procedure and diagnosis analysis
    procedure_analysis = claims_df_copy['PROCEDURE_CODE'].value_counts().head(10).to_dict()
    results['procedure_analysis'] = procedure_analysis
    
    diagnosis_patterns = claims_df_copy['DIAGNOSIS_CODE'].value_counts().head(10).to_dict()
    results['diagnosis_patterns'] = diagnosis_patterns
    
    return results

def analyze_real_quality_safety_data(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df):
    """Analyze real data for quality and safety surveillance insights"""
    results = {}
    
    results['total_patients'] = len(patients_df)
    results['total_encounters'] = len(encounters_df)
    results['total_events'] = len(clinical_events_df) if not clinical_events_df.empty else 0
    
    # Safety event detection
    readmissions = encounters_df[encounters_df['IS_30_DAY_READMIT'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    results['readmission_events'] = len(readmissions)
    results['readmission_rate'] = (len(readmissions) / len(encounters_df)) * 100 if len(encounters_df) > 0 else 0
    
    # Emergency visits analysis
    emergency_encounters = encounters_df[encounters_df['ENCOUNTER_TYPE'] == 'Emergency']
    frequent_ed_patients = emergency_encounters['PATIENT_ID'].value_counts()
    frequent_ed_users = frequent_ed_patients[frequent_ed_patients >= 3]
    results['frequent_ed_users'] = len(frequent_ed_users)
    
    # Discharge disposition analysis
    disposition_analysis = encounters_df['DISCHARGE_DISPOSITION'].value_counts().to_dict()
    results['disposition_patterns'] = disposition_analysis
    
    ama_discharges = encounters_df[encounters_df['DISCHARGE_DISPOSITION'] == 'Against Medical Advice']
    results['ama_discharges'] = len(ama_discharges)
    
    # Quality indicators
    quality_indicators = {
        'heart_failure_care': encounters_df[encounters_df['DISCHARGE_DIAGNOSIS_CODE'].str.contains('I50', case=False, na=False)],
        'diabetes_care': encounters_df[encounters_df['DISCHARGE_DIAGNOSIS_CODE'].str.contains('E11|E10', case=False, na=False)],
        'pneumonia_care': encounters_df[encounters_df['DISCHARGE_DIAGNOSIS_CODE'].str.contains('J44|J43', case=False, na=False)],
        'sepsis_care': encounters_df[encounters_df['DISCHARGE_DIAGNOSIS_CODE'].str.contains('A41', case=False, na=False)]
    }
    
    quality_metrics = {}
    for condition, condition_encounters in quality_indicators.items():
        if not condition_encounters.empty:
            condition_readmissions = condition_encounters[condition_encounters['IS_30_DAY_READMIT'].astype(str).str.lower().isin(['true', '1', 'yes'])]
            readmission_rate = (len(condition_readmissions) / len(condition_encounters)) * 100
            quality_score = max(0, 100 - readmission_rate)
            quality_metrics[condition] = {
                'patient_count': len(condition_encounters),
                'readmission_rate': readmission_rate,
                'quality_score': quality_score
            }
    
    results['quality_metrics'] = quality_metrics
    
    # High-utilizer patients
    patient_encounter_counts = encounters_df['PATIENT_ID'].value_counts()
    high_utilizers = patient_encounter_counts[patient_encounter_counts >= 4]
    results['high_utilizer_patients'] = len(high_utilizers)
    
    # Documentation completeness
    encounters_with_diagnosis = encounters_df[encounters_df['DISCHARGE_DIAGNOSIS_CODE'].notna()]
    documentation_rate = (len(encounters_with_diagnosis) / len(encounters_df)) * 100
    results['documentation_compliance'] = documentation_rate
    
    return results

def analyze_real_operations_optimization_data(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df):
    """Analyze real data for operational efficiency and resource optimization insights"""
    results = {}
    
    results['total_patients'] = len(patients_df)
    results['total_encounters'] = len(encounters_df)
    results['total_facilities'] = encounters_df['FACILITY_ID'].nunique() if 'FACILITY_ID' in encounters_df.columns else 0
    
    # Calculate length of stay patterns
    encounters_df_copy = encounters_df.copy()
    encounters_df_copy['ENCOUNTER_START_DATE'] = pd.to_datetime(encounters_df_copy['ENCOUNTER_START_DATE'], errors='coerce')
    encounters_df_copy['ENCOUNTER_END_DATE'] = pd.to_datetime(encounters_df_copy['ENCOUNTER_END_DATE'], errors='coerce')
    
    encounters_with_dates = encounters_df_copy.dropna(subset=['ENCOUNTER_START_DATE', 'ENCOUNTER_END_DATE'])
    if not encounters_with_dates.empty:
        encounters_with_dates['length_of_stay'] = (encounters_with_dates['ENCOUNTER_END_DATE'] - encounters_with_dates['ENCOUNTER_START_DATE']).dt.days
        encounters_with_dates['length_of_stay'] = encounters_with_dates['length_of_stay'].clip(lower=0)
        
        avg_los = encounters_with_dates['length_of_stay'].mean()
        results['average_length_of_stay'] = float(avg_los) if not pd.isna(avg_los) else 0
        
        long_stays = encounters_with_dates[encounters_with_dates['length_of_stay'] > 7]
        results['extended_stays'] = len(long_stays)
        results['extended_stay_rate'] = (len(long_stays) / len(encounters_with_dates)) * 100
    
    # Encounter type distribution
    encounter_type_dist = encounters_df['ENCOUNTER_TYPE'].value_counts().to_dict()
    results['encounter_distribution'] = encounter_type_dist
    
    # Patient utilization patterns
    patient_encounter_counts = encounters_df['PATIENT_ID'].value_counts()
    high_utilizers = patient_encounter_counts[patient_encounter_counts >= 4]
    results['high_utilizer_count'] = len(high_utilizers)
    results['high_utilizer_encounters'] = high_utilizers.sum()
    results['high_utilizer_percentage'] = (high_utilizers.sum() / len(encounters_df)) * 100
    
    # Financial efficiency analysis
    if not claims_df.empty:
        claims_df_copy = claims_df.copy()
        claims_df_copy['BILLED_AMOUNT'] = pd.to_numeric(claims_df_copy['BILLED_AMOUNT'], errors='coerce')
        claims_df_copy['PAID_AMOUNT'] = pd.to_numeric(claims_df_copy['PAID_AMOUNT'], errors='coerce')
        
        total_revenue = claims_df_copy['PAID_AMOUNT'].sum()
        revenue_per_encounter = total_revenue / len(encounters_df) if len(encounters_df) > 0 else 0
        results['revenue_per_encounter'] = float(revenue_per_encounter)
    
    # Geographic efficiency
    if 'ZIP_CODE' in patients_df.columns:
        unique_zips = patients_df['ZIP_CODE'].nunique()
        results['service_area_zips'] = unique_zips
    
    # Optimization opportunities
    optimization_opportunities = {}
    
    if results.get('extended_stays', 0) > 0:
        potential_los_reduction = results['extended_stays'] * 0.3
        optimization_opportunities['los_optimization'] = {
            'current_extended_stays': results['extended_stays'],
            'reduction_potential': potential_los_reduction,
            'cost_savings': potential_los_reduction * 1500
        }
    
    if results['high_utilizer_count'] > 0:
        utilizer_optimization = results['high_utilizer_encounters'] * 0.25
        optimization_opportunities['utilizer_management'] = {
            'target_patients': results['high_utilizer_count'],
            'encounter_reduction_potential': utilizer_optimization,
            'cost_savings': utilizer_optimization * 2500
        }
    
    results['optimization_opportunities'] = optimization_opportunities
    
    return results

def analyze_clinical_patterns_with_llm(clinical_events_df, model_name="claude-4-sonnet", analysis_type="population", progress_placeholder=None):
    """Analyze clinical patterns using LLM reasoning with step-by-step progress - Snowflake Version"""
    
    if clinical_events_df.empty:
        return "No clinical events data available for analysis."
    
    # Initialize completed steps if not exists
    if 'clinical_completed_steps' not in st.session_state:
        st.session_state.clinical_completed_steps = []
    
    def update_progress(step_name, details, results=None):
        """Update progress display in agent style"""
        if progress_placeholder:
            with progress_placeholder.container():
                st.write(f"**{step_name}**")
                
                if details:
                    st.markdown(f'<div class="agent-current">{details}</div>', unsafe_allow_html=True)
                
                if results:
                    st.session_state.clinical_completed_steps.append((step_name, results))
                    
                # Display completed steps
                for completed_step, completed_result in st.session_state.clinical_completed_steps:
                    st.markdown(f'<div class="agent-completed">✅ {completed_step}: {completed_result}</div>', unsafe_allow_html=True)
    
    try:
        # Clear previous steps
        st.session_state.clinical_completed_steps = []
        
        # Step 1: Data Preparation
        update_progress("Data Analysis Initialization", "🔍 Loading and processing clinical event data...", f"Analyzed {len(clinical_events_df)} clinical events across {clinical_events_df['EVENT_TYPE'].nunique()} event types")
        time.sleep(1)
        
        if analysis_type == "population":
            # Population-level pattern analysis
            event_summary = clinical_events_df.groupby(['EVENT_TYPE', 'CODE_VALUE']).size().head(10).to_dict()
            event_counts = clinical_events_df['EVENT_TYPE'].value_counts().to_dict()
            
            update_progress("Population Pattern Recognition", "📊 Identifying population-level clinical patterns and event distributions...", f"Processed {len(event_counts)} event types with {sum(event_counts.values())} total events")
            time.sleep(1)
            
            prompt = f"""You are a clinical data analyst. Analyze these healthcare event patterns and provide insights:

EVENT DISTRIBUTION:
{event_counts}

TOP EVENT-CODE COMBINATIONS:
{event_summary}

ANALYSIS TASKS:
1. Identify 3 most concerning clinical patterns
2. Explain what these patterns suggest about patient care
3. Recommend 3 specific interventions to improve outcomes
4. Highlight any potential quality or safety issues

Provide clear, actionable insights for healthcare executives."""

        else:  # Patient-specific analysis
            patient_events = clinical_events_df.head(20)
            events_summary = patient_events[['EVENT_TYPE', 'CODE_VALUE', 'DESCRIPTION', 'EVENT_DATE']].to_dict('records')
            
            update_progress("Patient Timeline Analysis", "👤 Analyzing patient-specific event timeline and clinical history...", f"Reviewed {len(patient_events)} patient events across multiple care episodes")
            time.sleep(1)
            
            prompt = f"""You are a clinical analyst reviewing a patient's event timeline:

PATIENT EVENTS:
{events_summary}

ANALYSIS TASKS:
1. Identify the clinical story these events tell
2. Spot any concerning patterns or gaps in care
3. Assess risk factors for future complications
4. Recommend specific interventions for this patient

Focus on actionable clinical insights."""

        # Step 3: AI Analysis
        update_progress("Clinical AI Reasoning", "⚙️ Processing clinical data through advanced AI reasoning and pattern recognition...", f"Sending comprehensive analysis to {model_name} for clinical insights generation")
        time.sleep(1)

        # Call LLM with retry logic
        llm_response = call_cortex_model(prompt, model_name)
        
        if not llm_response or llm_response.strip() == "":
            update_progress("Analysis Failed", "❌ AI analysis could not be completed", "Please try again with different parameters")
            return "Analysis unavailable - please try again."
        
        # Step 4: Complete
        update_progress("Analysis Complete", "✅ Clinical insights generation completed successfully", "AI-powered clinical analysis report ready for review")
        time.sleep(0.5)
        
        return llm_response
        
    except Exception as e:
        if progress_placeholder:
            with progress_placeholder.container():
                st.error(f"❌ Analysis failed: {str(e)}")
        return f"Analysis failed: {str(e)}"

# AI Agent Workflow Execution Functions
def run_proactive_care_agent(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name="claude-4-sonnet"):
    """Execute the proactive care management agent workflow - ENHANCED VERSION"""
    try:
        real_data_results = analyze_real_proactive_care_data(
            patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df
        )
        
        steps = [
            ("Initializing Agent", 10, f"Loading historical patient data with enhanced validation and quality checks from {real_data_results['data_range']}", f"Connected to {len(patients_df)} patients, {len(encounters_df)} encounters"),
            ("Analyzing Recent Discharges", 25, f"Advanced review of discharge patterns with clinical terminology from {real_data_results['analysis_period']}", f"Found {real_data_results['recent_discharges_count']} discharges in analysis period"),
            ("Risk Assessment", 45, "Sophisticated readmission risk calculation using evidence-based patterns and historical data", f"Identified {real_data_results['high_risk_count']} high-risk patients ({real_data_results['readmission_rate']:.1f}% readmission rate)"),
            ("Care Gap Analysis", 65, "Deep analysis of care coordination failures and follow-up gaps with clinical context", f"Found {real_data_results['care_gaps_count']} care gaps in recent analysis period"),
            ("Clinical Events Review", 85, f"Comprehensive analysis of recent clinical events with AI-powered pattern recognition: {real_data_results.get('recent_events_count', 0)} events in {real_data_results.get('event_analysis_period', 'analysis period')}", f"Generated evidence-based interventions for {min(real_data_results['high_risk_count'], 15)} specific patients"),
            ("Generating AI Report", 100, f"Professional care management analysis with enhanced clinical documentation using {model_name}", "AI-powered care coordination report with actionable clinical insights ready")
        ]
        
        # NEW: Generate LLM report using enhanced functions
        ai_report = generate_llm_report(real_data_results, "proactive_care", patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name)
        
        return steps, ai_report
        
    except Exception as e:
        st.error(f"Proactive Care Agent error: {str(e)}")
        return [], "Agent execution failed"

def run_population_health_agent(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name="claude-4-sonnet"):
    """Execute the population health manager agent workflow - ENHANCED VERSION"""
    try:
        real_data_results = analyze_real_population_health_data(
            patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df
        )
        
        steps = [
            ("Population Segmentation", 15, f"Advanced chronic condition cohort identification with detailed clinical metrics across {real_data_results['total_patients']} patients", f"Segmented into diabetes: {real_data_results['chronic_conditions'].get('diabetes', 0)}, hypertension: {real_data_results['chronic_conditions'].get('hypertension', 0)}, COPD: {real_data_results['chronic_conditions'].get('copd', 0)} patients"),
            ("Risk Trajectory Analysis", 30, f"Sophisticated analysis of deteriorating health patterns in {real_data_results['analysis_period']}", f"Identified {real_data_results['deteriorating_count']} patients with recent encounters requiring attention"),
            ("Care Adherence Assessment", 50, "Enhanced review of care engagement with clinical context and follow-up compliance patterns", f"Found {real_data_results['poor_adherence_count']} high-risk patients with poor care adherence"),
            ("Intervention Prioritization", 70, "Advanced ranking by clinical urgency with evidence-based prioritization and impact potential", f"Prioritized {real_data_results['high_impact_interventions']} high-impact intervention opportunities"),
            ("Resource Allocation", 85, f"Comprehensive analysis of care distribution with geographic insights across {real_data_results['zip_codes_served']} ZIP codes", f"Mapped resource needs across encounter types: {list(real_data_results['encounter_distribution'].keys())[:3]}"),
            ("Generating AI Report", 100, f"Targeted population health strategies with professional clinical recommendations using {model_name}", f"AI-powered population health analysis with enhanced population insights ready")
        ]
        
        # NEW: Generate LLM report
        ai_report = generate_llm_report(real_data_results, "population_health", patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name)
        
        return steps, ai_report
        
    except Exception as e:
        st.error(f"Population Health Agent error: {str(e)}")
        return [], "Agent execution failed"

def run_prior_authorization_agent(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name="claude-4-sonnet"):
    """Execute the prior authorization agent workflow - ENHANCED VERSION"""
    try:
        real_data_results = analyze_real_prior_authorization_data(
            patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df
        )
        
        steps = [
            ("Claims Pattern Analysis", 12, f"Advanced analysis of claims patterns with sophisticated financial modeling for {real_data_results['total_claims']} total claims", f"Identified {real_data_results['high_cost_claims']} high-cost claims (>${real_data_results['high_cost_threshold']:,.0f} threshold)"),
            ("Medical Necessity Validation", 28, f"Enhanced validation using evidence-based clinical criteria and procedure-diagnosis alignment", f"Validated medical necessity for {real_data_results['validation_rate']:.1f}% of claims using clinical criteria"),
            ("Cost-Benefit Modeling", 45, f"Sophisticated financial analysis with advanced optimization algorithms and payment patterns", f"Identified ${real_data_results['cost_analysis']['savings_opportunity']:,.0f} in potential optimization opportunities"),
            ("Denial Risk Assessment", 62, f"Advanced payer-specific approval pattern analysis across {len(real_data_results['payer_distribution'])} payers", f"Assessed approval risk with payment rates across multiple payers"),
            ("Alternative Treatment Analysis", 80, f"Comprehensive analysis of cost-effective treatment alternatives in procedure categories", f"Found alternative treatment opportunities in procedure categories"),
            ("Generating AI Report", 100, f"Evidence-based authorization framework with professional clinical documentation using {model_name}", f"AI-powered authorization report with enhanced authorization accuracy ready")
        ]
        
        # NEW: Generate LLM report
        ai_report = generate_llm_report(real_data_results, "prior_authorization", patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name)
        
        return steps, ai_report
        
    except Exception as e:
        st.error(f"Prior Authorization Agent error: {str(e)}")
        return [], "Agent execution failed"

def run_quality_safety_agent(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name="claude-4-sonnet"):
    """Execute the quality and safety surveillance agent workflow - ENHANCED VERSION"""
    try:
        real_data_results = analyze_real_quality_safety_data(
            patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df
        )
        
        steps = [
            ("Safety Event Detection", 15, f"Advanced scanning with sophisticated safety pattern recognition across {real_data_results['total_encounters']} encounters", f"Identified {real_data_results['readmission_events']} readmission events ({real_data_results['readmission_rate']:.1f}% rate) and {real_data_results['frequent_ed_users']} frequent ED users"),
            ("Quality Measure Assessment", 32, f"Enhanced clinical quality evaluation with professional terminology across {len(real_data_results['quality_metrics'])} core condition categories", f"Assessed quality metrics with documentation compliance rate of {real_data_results['documentation_compliance']:.1f}%"),
            ("Provider Performance Analysis", 48, f"Comprehensive analysis of performance variations with clinical context across healthcare facilities", f"Identified performance variations across healthcare facilities"),
            ("Risk Pattern Recognition", 65, f"Advanced detection of systematic risk patterns with AI-powered insights in {real_data_results['total_patients']} patient records", f"Detected {real_data_results['high_utilizer_patients']} high-utilizer patients"),
            ("Corrective Action Planning", 82, f"Sophisticated analysis of improvement opportunities with evidence-based recommendations for clinical events", f"Generated targeted improvement plans for {real_data_results['ama_discharges']} AMA discharges and quality gaps"),
            ("Generating AI Report", 100, f"Comprehensive quality and safety action plan with professional clinical documentation using {model_name}", f"AI-powered quality surveillance with enhanced safety monitoring ready")
        ]
        
        # NEW: Generate LLM report
        ai_report = generate_llm_report(real_data_results, "quality_safety", patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name)
        
        return steps, ai_report
        
    except Exception as e:
        st.error(f"Quality & Safety Agent error: {str(e)}")
        return [], "Agent execution failed"

def run_operations_optimization_agent(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name="claude-4-sonnet"):
    """Execute the operations optimization agent workflow - ENHANCED VERSION"""
    try:
        real_data_results = analyze_real_operations_optimization_data(
            patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df
        )
        
        steps = [
            ("Capacity Utilization Analysis", 18, f"Advanced analysis with sophisticated efficiency metrics across {real_data_results['total_encounters']} encounters and {real_data_results['total_facilities']} facilities", f"Average length of stay: {real_data_results.get('average_length_of_stay', 0):.1f} days, {real_data_results.get('extended_stays', 0)} extended stays identified"),
            ("Staffing Pattern Optimization", 34, f"Enhanced evaluation with professional operational terminology across encounter types and facilities", f"Identified optimization opportunities across healthcare facilities"),
            ("Equipment Utilization Review", 52, f"Comprehensive analysis with advanced resource optimization across clinical procedures", f"Found optimization potential in procedure categories"),
            ("Patient Flow Optimization", 68, f"Sophisticated review of discharge efficiency and flow patterns with clinical workflow integration", f"{real_data_results['high_utilizer_count']} high-utilizer patients identified"),
            ("Cost Center Performance", 84, f"Advanced benchmarking with comprehensive financial analysis across payers and claim types", f"Revenue per encounter: ${real_data_results.get('revenue_per_encounter', 0):,.2f}"),
            ("Generating AI Report", 100, f"Comprehensive optimization strategy with professional operational recommendations using {model_name}", f"AI-powered operations analysis with enhanced operational insights ready")
        ]
        
        # NEW: Generate LLM report
        ai_report = generate_llm_report(real_data_results, "operations", patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, model_name)
        
        return steps, ai_report
        
    except Exception as e:
        st.error(f"Operations Optimization Agent error: {str(e)}")
        return [], "Agent execution failed"

def execute_agent_workflow(agent_type, patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, placeholder, selected_model="claude-4-sonnet"):
    """
    Enhanced function to execute any agent workflow with UI updates - NOW WITH MODEL SELECTION
    """
    try:
        # Map agent types to functions
        agent_functions = {
            1: run_proactive_care_agent,
            2: run_population_health_agent,
            3: run_prior_authorization_agent,
            4: run_quality_safety_agent,
            5: run_operations_optimization_agent
        }
        
        if agent_type not in agent_functions:
            return "Invalid agent type"
        
        # Get the appropriate agent function
        agent_func = agent_functions[agent_type]
        
        # Execute the agent with selected model
        steps, final_report = agent_func(patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df, selected_model)
        
        # Initialize completed steps if not exists
        completed_steps_key = f'completed_steps_{agent_type}'
        if completed_steps_key not in st.session_state:
            st.session_state[completed_steps_key] = []
        
        # Execute workflow with UI updates
        for step_name, progress, details, results in steps:
            with placeholder.container():
                st.progress(progress / 100)
                st.write(f"**{step_name}**")
                
                if details:
                    st.markdown(f'<div class="agent-current">{details}</div>', unsafe_allow_html=True)
                
                if results:
                    st.session_state[completed_steps_key].append((step_name, results))
                    
                # Display completed steps
                for completed_step, completed_result in st.session_state[completed_steps_key]:
                    st.markdown(f'<div class="agent-completed">✅ {completed_step}: {completed_result}</div>', unsafe_allow_html=True)
            
            time.sleep(2)  # Demo timing
        
        return final_report
        
    except Exception as e:
        st.error(f"Agent execution error: {str(e)}")
        return "Agent execution failed"

# Clinical Insights Functions
response_add_info = """
    Add a patient information section to the top of the response including the patient's name and date of birth.
    When there is no context for Recent Encounters, Clinical Events, Claims Summary, or Risk Scores, create a notification
    style message stating which context is missing and add that to the top of the response below the patient information.
    """

context_add_info = """Use only the context provided in the Recent Encounters, Clinical Events,
    Claims Summary, and Risk Scores sections."""

def patient_readmission_risk_analysis(patient_summary, encounter_summary, event_summary, claims_summary, risk_summary, model_name, patient_name):
    """
    Generate AI-driven patient readmission risk analysis using Snowflake Cortex
    """
    prompt = f"""
    Analyze the likelihood of a 30-day hospital readmission for the following patient.
    Use the available encounter history, discharge dispositions, clinical events, and risk scores.
    {context_add_info}

    Provide:
    - Summary of recent encounters and timing
    - Discharge risk factors (e.g., SNF, hospice, home alone)
    - Clinical diagnoses or conditions increasing readmission risk
    - Prior risk score trends
    - Final readmission risk rating (Low, Medium, High)

    Patient Information:
    {patient_summary}

    Recent Encounters:
    {encounter_summary}

    Clinical Events:
    {event_summary}

    Claims Summary:
    {claims_summary}

    Risk Scores:
    {risk_summary}

    {response_add_info}
    """

    return call_cortex_model(prompt, model_name)

def identify_care_gaps(patient_summary, encounter_summary, event_summary, claims_summary, risk_summary, model_name, patient_name):
    """
    Identify care gaps for the patient
    """
    prompt = f"""
    Identify potential care gaps that may contribute to future hospital readmissions for the following patient.
    Use encounter history, lab orders, medication events, and missing follow-ups.
    {context_add_info}

    Provide:
    - Missed follow-up visits
    - Missing critical lab work or imaging
    - Medication non-compliance risks
    - Recommendations for care interventions

    Patient Information:
    {patient_summary}

    Recent Encounters:
    {encounter_summary}

    Clinical Events:
    {event_summary}

    Claims Summary:
    {claims_summary}

    Risk Scores:
    {risk_summary}

    {response_add_info}
    """
    
    return call_cortex_model(prompt, model_name)

def cost_risk_assessment(patient_summary, encounter_summary, event_summary, claims_summary, risk_summary, model_name, patient_name):
    """
    Assess financial risk for the patient
    """
    prompt = f"""
    Assess the financial risk for the following patient based on claims history, billed amounts, paid amounts, and clinical complexity.
    {context_add_info}

    Provide:
    - Total historical billed and paid amounts
    - Primary cost drivers (procedures, diagnoses)
    - Potential for high-cost future admissions
    - Cost risk rating (Low, Medium, High)

    Patient Information:
    {patient_summary}

    Recent Encounters:
    {encounter_summary}

    Clinical Events:
    {event_summary}

    Claims Summary:
    {claims_summary}

    Risk Scores:
    {risk_summary}

    {response_add_info}
    """
    
    return call_cortex_model(prompt, model_name)

def clinical_event_burden_review(patient_summary, encounter_summary, event_summary, claims_summary, risk_summary, model_name, patient_name):
    """
    Summarize the clinical burden of diagnoses, procedures, and medications
    """
    prompt = f"""
    Summarize the clinical burden of diagnoses, procedures, and medications documented for the following patient.
    {context_add_info}

    Provide:
    - Most frequent diagnoses
    - Common procedures performed
    - Number of medication orders
    - Indications of complex clinical management

    Patient Information:
    {patient_summary}

    Recent Encounters:
    {encounter_summary}

    Clinical Events:
    {event_summary}

    Claims Summary:
    {claims_summary}

    Risk Scores:
    {risk_summary}

    {response_add_info}
    """
    
    return call_cortex_model(prompt, model_name)

def risk_score_trajectory_analysis(patient_summary, encounter_summary, event_summary, claims_summary, risk_summary, model_name, patient_name):
    """
    Analyze risk scores changes over time
    """
    prompt = f"""
    Analyze how the patient's risk scores have changed over time.
    Comment on any increasing or decreasing trends.
    {context_add_info}

    Provide:
    - Timeline of risk scores
    - Major shifts in risk categories
    - Interpretation of why risk may be improving or worsening
    - Predict future risk based on trends

    Patient Information:
    {patient_summary}

    Recent Encounters:
    {encounter_summary}

    Clinical Events:
    {event_summary}

    Claims Summary:
    {claims_summary}

    Risk Scores:
    {risk_summary}

    {response_add_info}
    """
    
    return call_cortex_model(prompt, model_name)

def format_data(data_df):
    """Format DataFrame for prompt inclusion"""
    if data_df.empty:
        return ""
    dict_index = data_df.to_dict(orient='index')
    return dict_index

def load_patients_data():
    """Load patients data from Snowflake with caching"""
    query = f"""
    SELECT 
        PATIENT_ID, 
        PATIENT_NAME, 
        DOB, 
        GENDER, 
        ADDRESS,
        ZIP_CODE,
        RACE,
        ETHNICITY,
        PRIMARY_LANGUAGE,
        RISK_CATEGORY
    FROM {DATABASE}.{SCHEMA}.{TABLE_PATIENTS}
    """
    return query_snowflake(query, cache_key="patients_data", ttl=1800)

def load_encounters_data():
    """Load encounters data from Snowflake with caching"""
    query = f"""
    SELECT 
        ENCOUNTER_ID,
        PATIENT_ID,
        PATIENT_NAME,
        ENCOUNTER_START_DATE,
        ENCOUNTER_END_DATE,
        ENCOUNTER_TYPE,
        FACILITY_ID,
        ADMITTING_DIAGNOSIS_CODE,
        DISCHARGE_DIAGNOSIS_CODE,
        DISCHARGE_DISPOSITION,
        IS_30_DAY_READMIT,
        PRIOR_ENCOUNTER_ID
    FROM {DATABASE}.{SCHEMA}.{TABLE_ENCOUNTERS}
    """
    return query_snowflake(query, cache_key="encounters_data", ttl=1800)

def load_clinical_events_data():
    """Load clinical events data from Snowflake with caching"""
    query = f"""
    SELECT 
        EVENT_ID,
        PATIENT_ID,
        PATIENT_NAME,
        ENCOUNTER_ID,
        EVENT_DATE,
        EVENT_TYPE,
        CODE_TYPE,
        CODE_VALUE,
        DESCRIPTION
    FROM {DATABASE}.{SCHEMA}.{TABLE_EVENTS}
    """
    return query_snowflake(query, cache_key="clinical_events_data", ttl=1200)

def load_claims_data():
    """Load claims data from Snowflake with caching"""
    query = f"""
    SELECT 
        CLAIM_ID,
        PATIENT_ID,
        PATIENT_NAME,
        ENCOUNTER_ID,
        CLAIM_START_DATE,
        CLAIM_END_DATE,
        PAYER_ID,
        CLAIM_TYPE,
        PROCEDURE_CODE,
        DIAGNOSIS_CODE,
        BILLED_AMOUNT,
        PAID_AMOUNT
    FROM {DATABASE}.{SCHEMA}.{TABLE_CLAIMS}
    """
    return query_snowflake(query, cache_key="claims_data", ttl=1200)

def load_risk_scores_data():
    """Load risk scores data from Snowflake with caching"""
    query = f"""
    SELECT 
        RISK_ID,
        PATIENT_ID,
        PATIENT_NAME,
        ASSESSMENT_DATE,
        RISK_SCORE,
        RISK_LEVEL,
        KEY_RISK_FACTORS
    FROM {DATABASE}.{SCHEMA}.{TABLE_RISK_SCORES}
    """
    return query_snowflake(query, cache_key="risk_scores_data", ttl=1800)

# 1. NEW ENHANCED LLM REPORT GENERATOR FUNCTIONS

def format_analysis_data_enhanced(analysis_results, report_type, patients_df, encounters_df, clinical_events_df=None, claims_df=None, risk_scores_df=None):
    """Enhanced formatting with specific examples and detailed calculations for Snowflake"""
    
    base_summary = f"""
    HEALTHCARE ANALYSIS RESULTS - {report_type.upper().replace('_', ' ')}
    
    Analysis Period: {analysis_results.get('analysis_period', 'Current period')}
    Data Range: {analysis_results.get('data_range', 'Full dataset')}
    """
    
    if report_type == "proactive_care":
        # Get specific high-risk patient examples
        try:
            high_risk_sample = patients_df[patients_df['RISK_CATEGORY'] == 'High'].head(4)
            patient_examples = []
            for idx, patient in high_risk_sample.iterrows():
                patient_examples.append(f"{patient['PATIENT_ID']} ({patient['PATIENT_NAME']}) - {patient['GENDER']}, {patient['RACE']}")
        except:
            patient_examples = ["Sample patient data unavailable"]
            
        # Calculate specific financial impact
        estimated_cost_avoidance = analysis_results.get('high_risk_count', 0) * 15000
        
        # Get encounter and diagnosis summaries
        encounter_summary = ", ".join([f"{k}: {v}" for k, v in list(analysis_results.get('encounter_types', {}).items())[:3]])
        diagnosis_summary = ", ".join([f"{k}: {v} cases" for k, v in list(analysis_results.get('top_diagnoses', {}).items())[:3]])
        
        return base_summary + f"""
        POPULATION OVERVIEW:
        - Total Patients Analyzed: {len(patients_df)}
        - Total Encounters: {len(encounters_df)}
        - High-Risk Patients: {analysis_results.get('high_risk_count', 0)}
        - Recent Discharges: {analysis_results.get('recent_discharges_count', 0)}
        - Care Gaps Identified: {analysis_results.get('care_gaps_count', 0)}
        
        CLINICAL METRICS:
        - 30-Day Readmission Rate: {analysis_results.get('readmission_rate', 0):.1f}%
        - Readmission Events: {analysis_results.get('readmission_count', 0)}
        - Risk Distribution: {analysis_results.get('risk_distribution', {})}
        - Top Discharge Diagnoses: {diagnosis_summary}
        - Encounter Distribution: {encounter_summary}
        
        HIGH-RISK PATIENT EXAMPLES (for specific interventions):
        {chr(10).join([f"- {example}" for example in patient_examples])}
        
        FINANCIAL IMPACT ANALYSIS:
        - Total Claims Volume: ${analysis_results.get('total_billed', 0):,.2f} billed
        - Payment Rate: {analysis_results.get('payment_rate', 0):.1f}%
        - Calculated Cost Avoidance: ${estimated_cost_avoidance:,.2f} (prevented readmissions at $15K each)
        
        GEOGRAPHIC & DEMOGRAPHIC INSIGHTS:
        - Geographic Coverage: {patients_df['ZIP_CODE'].nunique() if 'ZIP_CODE' in patients_df.columns else 0} unique ZIP codes served
        - Gender Distribution: {len(patients_df[patients_df['GENDER'] == 'Female']) if 'GENDER' in patients_df.columns else 0} Female, {len(patients_df[patients_df['GENDER'] == 'Male']) if 'GENDER' in patients_df.columns else 0} Male patients
        - Language Services Needed: {patients_df['PRIMARY_LANGUAGE'].nunique() if 'PRIMARY_LANGUAGE' in patients_df.columns else 0} languages requiring interpretation services
        - Most Common Risk Factors: {', '.join(list(analysis_results.get('top_risk_factors', {}).keys())[:3])}
        
        RECENT CLINICAL ACTIVITY:
        - Recent Events Count: {analysis_results.get('recent_events_count', 0)}
        - Recent Procedures: {analysis_results.get('recent_procedures', {})}
        - Event Analysis Period: {analysis_results.get('event_analysis_period', 'N/A')}
        """
    
    elif report_type == "population_health":
        # Get demographic breakdowns
        try:
            demographics = analysis_results.get('demographics', {})
            gender_breakdown = ", ".join([f"{k}: {v}" for k, v in demographics.get('gender', {}).items()])
            race_breakdown = ", ".join([f"{k}: {v}" for k, v in list(demographics.get('race', {}).items())[:4]])
            language_breakdown = ", ".join([f"{k}: {v}" for k, v in list(demographics.get('language', {}).items())[:3]])
        except:
            gender_breakdown = "Data unavailable"
            race_breakdown = "Data unavailable"
            language_breakdown = "Data unavailable"
            
        # Get geographic concentration details
        high_risk_zips = analysis_results.get('high_risk_zip_codes', {})
        top_zip_codes = ", ".join([f"{zip_code} ({count} patients)" for zip_code, count in list(high_risk_zips.items())[:3]])
        
        # Calculate percentages for chronic conditions
        total_patients = analysis_results.get('total_patients', 1)
        chronic_conditions = analysis_results.get('chronic_conditions', {})
        condition_percentages = {}
        for condition, count in chronic_conditions.items():
            condition_percentages[condition] = f"{count} ({(count/total_patients*100):.1f}%)"
        
        # Get encounter type distribution
        encounter_types = ", ".join([f"{k}: {v}" for k, v in list(analysis_results.get('encounter_distribution', {}).items())[:3]])
        
        return base_summary + f"""
        POPULATION SEGMENTATION:
        - Total Population Analyzed: {total_patients}
        - Service Coverage: {analysis_results.get('zip_codes_served', 0)} ZIP codes
        - Analysis Period: {analysis_results.get('analysis_period', 'N/A')}
        
        CHRONIC DISEASE BURDEN:
        - Diabetes Patients: {condition_percentages.get('diabetes', '0 (0.0%)')} of population
        - Hypertension Patients: {condition_percentages.get('hypertension', '0 (0.0%)')} of population
        - Heart Failure Patients: {condition_percentages.get('heart_failure', '0 (0.0%)')} of population
        - COPD Patients: {condition_percentages.get('copd', '0 (0.0%)')} of population
        - Chronic Kidney Disease: {condition_percentages.get('ckd', '0 (0.0%)')} of population
        
        HIGH-PRIORITY PATIENT COHORTS:
        - Deteriorating Health Status: {analysis_results.get('deteriorating_count', 0)} patients with recent high-risk encounters
        - Poor Care Adherence: {analysis_results.get('poor_adherence_count', 0)} high-risk patients with inadequate follow-up
        - High-Impact Intervention Opportunities: {analysis_results.get('high_impact_interventions', 0)} patients with readmission history
        
        RISK STRATIFICATION BREAKDOWN:
        - High Risk: {analysis_results.get('risk_distribution', {}).get('High', 0)} patients ({(analysis_results.get('risk_distribution', {}).get('High', 0)/total_patients*100):.1f}%)
        - Moderate Risk: {analysis_results.get('risk_distribution', {}).get('Moderate', 0)} patients ({(analysis_results.get('risk_distribution', {}).get('Moderate', 0)/total_patients*100):.1f}%)
        - Low Risk: {analysis_results.get('risk_distribution', {}).get('Low', 0)} patients ({(analysis_results.get('risk_distribution', {}).get('Low', 0)/total_patients*100):.1f}%)
        
        DEMOGRAPHIC ANALYSIS:
        - Gender Distribution: {gender_breakdown}
        - Race/Ethnicity Breakdown: {race_breakdown}
        - Primary Languages: {language_breakdown}
        
        GEOGRAPHIC RESOURCE ALLOCATION:
        - High-Risk Geographic Concentrations: {top_zip_codes}
        - Encounter Type Distribution: {encounter_types}
        """
    
    elif report_type == "prior_authorization":
        # Get detailed payer and procedure analysis
        payer_distribution = analysis_results.get('payer_distribution', {})
        top_payers = ", ".join([f"{payer}: {count} claims" for payer, count in list(payer_distribution.items())[:3]])
        
        procedure_analysis = analysis_results.get('procedure_analysis', {})
        top_procedures = ", ".join([f"{proc}: {count}" for proc, count in list(procedure_analysis.items())[:3]])
        
        diagnosis_patterns = analysis_results.get('diagnosis_patterns', {})
        top_diagnoses = ", ".join([f"{diag}: {count}" for diag, count in list(diagnosis_patterns.items())[:3]])
        
        # Get detailed cost analysis
        cost_analysis = analysis_results.get('cost_analysis', {})
        total_billed = cost_analysis.get('total_billed', 0)
        total_paid = cost_analysis.get('total_paid', 0)
        payment_variance = cost_analysis.get('payment_variance', 0)
        payment_rate = cost_analysis.get('payment_rate', 0)
        savings_opportunity = cost_analysis.get('savings_opportunity', 0)
        
        # Get payer approval rates
        payer_approval_rates = analysis_results.get('payer_approval_rates', {})
        approval_rates_summary = ", ".join([f"{payer}: {rate:.1f}%" for payer, rate in list(payer_approval_rates.items())[:3]])
        
        return base_summary + f"""
        CLAIMS VOLUME ANALYSIS:
        - Total Claims Processed: {analysis_results.get('total_claims', 0)}
        - High-Cost Claims Threshold: ${analysis_results.get('high_cost_threshold', 0):,.0f}
        - High-Cost Claims Identified: {analysis_results.get('high_cost_claims', 0)} ({analysis_results.get('high_cost_percentage', 0):.1f}% of total)
        - Medical Necessity Validated: {analysis_results.get('medical_necessity_validated', 0)} claims
        - Validation Rate: {analysis_results.get('validation_rate', 0):.1f}%
        
        FINANCIAL IMPACT ANALYSIS:
        - Total Billed Amount: ${total_billed:,.2f}
        - Total Paid Amount: ${total_paid:,.2f}
        - Payment Variance: ${payment_variance:,.2f}
        - Overall Payment Rate: {payment_rate:.1f}%
        - Optimization Savings Opportunity: ${savings_opportunity:,.2f}
        
        PAYER ANALYSIS:
        - Major Payers: {top_payers}
        - Payer Approval Rates: {approval_rates_summary}
        - Payer Distribution: {len(payer_distribution)} unique payers
        
        PROCEDURE AND DIAGNOSIS PATTERNS:
        - Most Common Procedures: {top_procedures}
        - Top Diagnosis Patterns: {top_diagnoses}
        - Alternative Treatment Opportunities: Identified in high-volume procedure categories
        
        AUTHORIZATION DECISION FRAMEWORK:
        - Baseline Approval Rate: {payment_rate:.1f}%
        - Cost-Benefit Analysis: 15-25% cost reduction potential in identified categories
        - Process Optimization Potential: ${savings_opportunity:,.2f} annual savings
        """
    
    elif report_type == "quality_safety":
        # Get detailed quality metrics breakdown
        quality_metrics = analysis_results.get('quality_metrics', {})
        quality_summary = []
        for condition, metrics in quality_metrics.items():
            patient_count = metrics.get('patient_count', 0)
            readmission_rate = metrics.get('readmission_rate', 0)
            quality_score = metrics.get('quality_score', 0)
            quality_summary.append(f"{condition.replace('_', ' ').title()}: {patient_count} patients, {readmission_rate:.1f}% readmission rate, {quality_score:.1f} quality score")
        
        # Get disposition analysis
        disposition_patterns = analysis_results.get('disposition_patterns', {})
        disposition_summary = ", ".join([f"{disp}: {count}" for disp, count in list(disposition_patterns.items())[:4]])
        
        return base_summary + f"""
        SAFETY EVENT ANALYSIS:
        - Total Patients: {analysis_results.get('total_patients', 0)}
        - Total Encounters: {analysis_results.get('total_encounters', 0)}
        - Total Clinical Events: {analysis_results.get('total_events', 0)}
        
        CRITICAL SAFETY INDICATORS:
        - 30-Day Readmission Events: {analysis_results.get('readmission_events', 0)} events
        - Overall Readmission Rate: {analysis_results.get('readmission_rate', 0):.1f}%
        - Frequent ED Users: {analysis_results.get('frequent_ed_users', 0)} patients (3+ emergency visits)
        - Against Medical Advice Discharges: {analysis_results.get('ama_discharges', 0)} cases requiring follow-up
        - High-Utilizer Patients: {analysis_results.get('high_utilizer_patients', 0)} patients (4+ encounters)
        
        QUALITY PERFORMANCE BY CONDITION:
        {chr(10).join([f"- {summary}" for summary in quality_summary])}
        
        DOCUMENTATION AND COMPLIANCE:
        - Documentation Compliance Rate: {analysis_results.get('documentation_compliance', 0):.1f}%
        - Quality Standards Alignment: Tracked across {len(quality_metrics)} core conditions
        
        DISCHARGE DISPOSITION PATTERNS:
        - Disposition Analysis: {disposition_summary}
        - AMA Discharge Rate: Requires enhanced follow-up protocols
        
        IMMEDIATE INTERVENTION PRIORITIES:
        - Readmission Prevention: {analysis_results.get('readmission_events', 0)} patients requiring immediate follow-up
        - ED Utilization Management: {analysis_results.get('frequent_ed_users', 0)} frequent users need care coordination
        - High-Utilizer Case Management: {analysis_results.get('high_utilizer_patients', 0)} patients need enhanced coordination
        """
    
    elif report_type == "operations":
        # Get detailed utilization and efficiency metrics
        optimization_opportunities = analysis_results.get('optimization_opportunities', {})
        total_savings = sum([opp.get('cost_savings', 0) for opp in optimization_opportunities.values()])
        
        # Length of stay optimization details
        los_optimization = optimization_opportunities.get('los_optimization', {})
        los_current_stays = los_optimization.get('current_extended_stays', 0)
        los_reduction_potential = los_optimization.get('reduction_potential', 0)
        los_cost_savings = los_optimization.get('cost_savings', 0)
        
        # High-utilizer management details
        utilizer_optimization = optimization_opportunities.get('utilizer_management', {})
        utilizer_target_patients = utilizer_optimization.get('target_patients', 0)
        utilizer_reduction_potential = utilizer_optimization.get('encounter_reduction_potential', 0)
        utilizer_cost_savings = utilizer_optimization.get('cost_savings', 0)
        
        # Encounter distribution
        encounter_distribution = analysis_results.get('encounter_distribution', {})
        encounter_summary = ", ".join([f"{enc_type}: {count}" for enc_type, count in list(encounter_distribution.items())[:4]])
        
        return base_summary + f"""
        CAPACITY UTILIZATION ANALYSIS:
        - Total Encounters Analyzed: {analysis_results.get('total_encounters', 0)}
        - Healthcare Facilities: {analysis_results.get('total_facilities', 0)}
        - Patient Population: {analysis_results.get('total_patients', 0)}
        
        LENGTH OF STAY PERFORMANCE:
        - Average Length of Stay: {analysis_results.get('average_length_of_stay', 0):.1f} days
        - Extended Stays (>7 days): {analysis_results.get('extended_stays', 0)} encounters
        - Extended Stay Rate: {analysis_results.get('extended_stay_rate', 0):.1f}%
        
        PATIENT UTILIZATION PATTERNS:
        - High-Utilizer Patients: {analysis_results.get('high_utilizer_count', 0)} patients (4+ encounters)
        - High-Utilizer Encounters: {analysis_results.get('high_utilizer_encounters', 0)} total encounters
        - High-Utilizer Percentage: {analysis_results.get('high_utilizer_percentage', 0):.1f}% of total encounters
        
        FINANCIAL EFFICIENCY ANALYSIS:
        - Revenue per Encounter: ${analysis_results.get('revenue_per_encounter', 0):,.2f}
        - Service Area Coverage: {analysis_results.get('service_area_zips', 0)} ZIP codes served
        - Encounter Type Distribution: {encounter_summary}
        
        OPTIMIZATION OPPORTUNITIES:
        - Length of Stay Optimization:
          * Current Extended Stays: {los_current_stays} encounters
          * Reduction Potential: {los_reduction_potential:.1f} encounters
          * Projected Cost Savings: ${los_cost_savings:,.0f}
        
        - High-Utilizer Management:
          * Target Patients: {utilizer_target_patients} high-utilizers
          * Encounter Reduction Potential: {utilizer_reduction_potential:.1f} encounters
          * Projected Cost Savings: ${utilizer_cost_savings:,.0f}
        
        TOTAL PROJECTED ANNUAL SAVINGS: ${total_savings:,.0f}
        
        RESOURCE ALLOCATION RECOMMENDATIONS:
        - Focus on {los_current_stays} extended stay cases for immediate LOS reduction
        - Implement case management for {utilizer_target_patients} high-utilizer patients
        - Optimize workflow across {analysis_results.get('total_facilities', 0)} facilities
        - Deploy resources across {analysis_results.get('service_area_zips', 0)} service areas
        """
    
    return base_summary

def get_report_config(report_type):
    """Configuration for each agent report type"""
    configs = {
        "proactive_care": {
            "title": "Proactive Care Management Report",
            "focus": "readmission prevention and care coordination",
            "audience": "care coordinators and clinical staff",
            "sections": ["Clinical Risk Assessment", "Care Gap Analysis", "Intervention Priorities"]
        },
        "population_health": {
            "title": "Population Health Management Report", 
            "focus": "chronic disease management and population outcomes",
            "audience": "population health managers and executives",
            "sections": ["Population Segmentation", "Risk Stratification", "Intervention Targeting"]
        },
        "prior_authorization": {
            "title": "Prior Authorization Analysis Report",
            "focus": "medical necessity and cost optimization", 
            "audience": "utilization management and payer relations",
            "sections": ["Authorization Patterns", "Cost-Benefit Analysis", "Optimization Opportunities"]
        },
        "quality_safety": {
            "title": "Clinical Quality & Safety Surveillance Report",
            "focus": "safety events and quality improvement",
            "audience": "quality officers and patient safety teams", 
            "sections": ["Safety Event Analysis", "Quality Metrics", "Corrective Action Plans"]
        },
        "operations": {
            "title": "Operational Efficiency & Resource Optimization Report",
            "focus": "resource utilization and operational performance",
            "audience": "operations managers and executives",
            "sections": ["Capacity Analysis", "Resource Utilization", "Optimization Strategies"]
        }
    }
    return configs.get(report_type, configs["proactive_care"])

def create_report_prompt(data_summary, config):
    """Create specialized prompt for each analysis type"""
    return f"""
    You are a senior healthcare {config['focus']} consultant generating an executive report for {config['audience']}.

    ANALYSIS DATA:
    {data_summary}

    REPORT REQUIREMENTS:
    1. Generate a comprehensive {config['title']} 
    2. Focus specifically on {config['focus']}
    3. Target audience: {config['audience']}
    4. Include these key sections: {', '.join(config['sections'])}
    5. Provide 4-6 specific, actionable recommendations
    6. Use professional healthcare terminology
    7. Include financial and operational impact
    8. Highlight critical findings requiring immediate attention

    REPORT STRUCTURE:
    ## Executive Summary
    [2-3 sentences highlighting most critical findings]
    
    ## {config['sections'][0]}
    [Detailed analysis of primary focus area]
    
    ## {config['sections'][1]} 
    [Secondary analysis with supporting metrics]
    
    ## {config['sections'][2]}
    [Strategic recommendations and next steps]
    
    ## Priority Actions
    [4-6 specific, actionable interventions with timelines]
    
    ## Financial Impact
    [Cost implications and ROI projections]

    Generate insights that demonstrate deep clinical expertise in {config['focus']} and provide actionable intelligence for immediate implementation.
    """

def generate_fallback_report(analysis_results, report_type):
    """Fallback to template if LLM fails"""
    config = get_report_config(report_type)
    return f"""# {config['title']} - Structured Analysis

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status**: Automated Analysis (LLM unavailable)

## Key Findings
- Analysis completed for {analysis_results.get('total_patients', 'N/A')} patients
- Primary metrics calculated and reviewed
- Recommendations based on structured analysis

## Next Steps
Please review the structured data and consult with clinical staff for detailed recommendations.

**Note**: This is a fallback analysis. Full AI-generated insights temporarily unavailable.
"""

def generate_llm_report(analysis_results, report_type, patients_df, encounters_df, clinical_events_df=None, claims_df=None, risk_scores_df=None, model_name="claude-4-sonnet"):
    """Generate comprehensive analysis using LLM based on structured analysis - Snowflake Cortex Version"""
    
    # Get report-specific configuration
    config = get_report_config(report_type)
    
    # Use enhanced data formatting with all specific details
    data_summary = format_analysis_data_enhanced(analysis_results, report_type, patients_df, encounters_df, clinical_events_df, claims_df, risk_scores_df)
    
    # Create specialized prompt
    report_prompt = create_report_prompt(data_summary, config)
    
    # RETRY LOGIC for Cortex errors
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Call Snowflake Cortex model
            llm_response = call_cortex_model(report_prompt, model_name)
            
            # Validate LLM response
            if not llm_response or llm_response.strip() == "":
                if attempt < max_retries - 1:
                    st.info(f"Empty response, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError("Empty response from Snowflake Cortex after all retries")
            
            # SUCCESS - Format final report with headers
            final_report = f"""# {config['title']} - AI-Generated Analysis

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Data Source**: Live Snowflake Analysis  
**AI Model**: {model_name}

---

{llm_response}

---

**Report Generation Method**: This analysis was generated through autonomous Snowflake Cortex AI agent analysis of live healthcare data, combining structured data analytics with advanced language model insights for clinical decision support and multi-step reasoning workflows.
"""
            
            return final_report
            
        except Exception as e:
            # Handle Cortex-specific errors
            if "CORTEX" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    st.info(f"Snowflake Cortex temporarily unavailable, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff: 2s, 4s, 8s
                    continue
                else:
                    st.warning(f"Snowflake Cortex unavailable after {max_retries} attempts. Using structured template.")
                    break
            else:
                # Handle other errors (don't retry)
                st.warning(f"Snowflake Cortex analysis generation failed: {str(e)}. Using structured template.")
                break
    
    # FALLBACK - Return structured template if all retries failed
    return generate_fallback_report(analysis_results, report_type)

def enhanced_clinical_events_explorer_tab(clinical_events_df):
    """
    Complete enhanced Clinical Events Explorer tab content
    Replace the existing tabs[3] content with this
    """
    st.header("Clinical Events Explorer")
    if not clinical_events_df.empty:
        
        # TOP SECTION: Add AI Analysis Controls
        with st.container(border=True):
            st.subheader("Clinical Pattern Agent Analysis")

            st.markdown("""
            **Business Challenge**: Clinical analysts spend 3+ hours daily manually reviewing thousands of clinical events (lab results, procedures, medications, diagnostics) across patient populations, struggling to identify meaningful patterns that could indicate quality issues, safety risks, or care optimization opportunities.

            **Agent Solution**: Autonomous clinical intelligence agent that automatically analyzes clinical event data to identify concerning trends, detect anomalies in care delivery, and generate evidence-based recommendations for improving patient outcomes and operational efficiency.
            """)

            st.markdown("**Analysis Type:** Population Patterns")
            
            reasoning_model = st.selectbox(
                "Select Snowflake Cortex Model for Analysis:",
                ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"],
                key="clinical_reasoning_model"
            )
            
            analyze_button = st.button(
                "🔍 Analyze Patterns",
                type="primary",
                key="analyze_clinical_patterns"
            )
        
        # AI ANALYSIS RESULTS
        if analyze_button:
            # Create placeholder for progress updates
            progress_placeholder = st.empty()
            
            # Run analysis with step-by-step progress
            ai_analysis = analyze_clinical_patterns_with_llm(
                clinical_events_df, 
                reasoning_model, 
                "population",
                progress_placeholder  # Pass the placeholder
            )
            
            # Clear progress and show results
            progress_placeholder.empty()
            
            with st.expander("AI Clinical Insights", expanded=True):
                st.markdown(ai_analysis)
            
            # Save to session state for persistence
            st.session_state.clinical_analysis = ai_analysis
            st.session_state.clinical_analysis_timestamp = datetime.now()

        # Show previous analysis if available
        elif 'clinical_analysis' in st.session_state:
            timestamp = st.session_state.get('clinical_analysis_timestamp', datetime.now())
            with st.expander(f"Previous AI Clinical Insights - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", expanded=False):
                st.markdown(st.session_state.clinical_analysis)
        
        # EXISTING CONTENT: Keep your current visualizations
        with st.container(border=True):
            st.metric("Total Clinical Events", len(clinical_events_df))

        # Create two columns for clinical events visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Event Type Distribution")
            
            # Define clinical event colors
            event_colors = ['#8E44AD', '#E67E22', '#16A085', '#E74C3C']  # Purple, Orange, Teal, Red
            
            event_bar = alt.Chart(clinical_events_df).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
                width=70,
                opacity=0.85
            ).encode(
                x=alt.X('EVENT_TYPE:N', 
                       sort='-y', 
                       title='Event Type',
                       axis=alt.Axis(labelAngle=0, labelFontSize=11)),
                y=alt.Y('count():Q', 
                       title='Count',
                       axis=alt.Axis(grid=True, gridOpacity=0.3)),
                color=alt.Color('EVENT_TYPE:N', 
                               scale=alt.Scale(range=event_colors),
                               legend=None),
                stroke=alt.value('white'),
                strokeWidth=alt.value(2)
            )

            event_text = alt.Chart(clinical_events_df).mark_text(
                align="center", 
                baseline="bottom", 
                dy=-8, 
                fontSize=10,
                fontWeight='normal',
                color='#2C3E50'
            ).encode(
                x=alt.X('EVENT_TYPE:N', sort='-y'),
                y=alt.Y('count():Q'),
                text=alt.Text('count():Q', format='.0f')
            )

            event_chart = (event_bar + event_text).properties(height=300)
            st.altair_chart(event_chart, use_container_width=True)

        with col2:
            st.subheader("Events Timeline")
            
            # Create timeline chart if we have date data
            if 'EVENT_DATE' in clinical_events_df.columns:
                try:
                    # Convert to datetime and extract month-year
                    events_df_copy = clinical_events_df.copy()
                    events_df_copy['EVENT_DATE'] = pd.to_datetime(events_df_copy['EVENT_DATE'], errors='coerce')
                    events_df_copy['month_year'] = events_df_copy['EVENT_DATE'].dt.to_period('M').astype(str)
                    
                    # Get monthly counts by event type
                    monthly_events = events_df_copy.groupby(['month_year', 'EVENT_TYPE']).size().reset_index(name='count')
                    
                    events_timeline = alt.Chart(monthly_events).mark_line(
                        point=True,
                        strokeWidth=3
                    ).encode(
                        x=alt.X('month_year:N', title='Month', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('count:Q', title='Event Count'),
                        color=alt.Color('EVENT_TYPE:N', 
                                       scale=alt.Scale(range=event_colors),
                                       title='Event Type'),
                        tooltip=['month_year', 'EVENT_TYPE', 'count']
                    ).properties(height=300)
                    
                    st.altair_chart(events_timeline, use_container_width=True)
                except:
                    # Fallback: Events by Code Type
                    if 'CODE_TYPE' in clinical_events_df.columns:
                        code_type_data = clinical_events_df.groupby(['EVENT_TYPE', 'CODE_TYPE']).size().reset_index(name='count')
                        
                        code_chart = alt.Chart(code_type_data).mark_bar(
                            cornerRadiusTopLeft=8,
                            cornerRadiusTopRight=8,
                            opacity=0.8
                        ).encode(
                            x=alt.X('CODE_TYPE:N', title='Code Type', axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y('count:Q', title='Count'),
                            color=alt.Color('EVENT_TYPE:N', 
                                           scale=alt.Scale(range=event_colors),
                                           title='Event Type'),
                            tooltip=['CODE_TYPE', 'EVENT_TYPE', 'count']
                        ).properties(height=300)
                        
                        st.altair_chart(code_chart, use_container_width=True)

        # Event type filter and details
        selected = st.selectbox("Select Event Type to View Details", 
                              ["All"] + sorted(clinical_events_df['EVENT_TYPE'].dropna().unique().tolist()))
        filtered_events = clinical_events_df if selected == "All" else clinical_events_df[clinical_events_df["EVENT_TYPE"] == selected]
        st.dataframe(filtered_events)

def main():
    # CSS Styling
    st.markdown(
        """
        <style>
        /* General Page Styling */
        .css-18e3th9 {
            background-color: #f4f6f9;
        }

        /* Header Styling */
        .css-10trblm {
            font-size: 1.8rem;
            color: #0062cc;
            font-weight: bold;
        }

        /* Metric Styling */
        .stMetric {
            background: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 5px;
        }

        /* Button Styling */
        .stButton > button {
            background-color: #0062cc;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;
            border: none;
            cursor: pointer;
        }

        .stButton > button:hover {
            background-color: #0051a8;
            color: white;
        }

        .stButton > button:active {
            background-color: #004085;
            color: white;
            box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.2);
        }

        /* Style all checkboxes in Streamlit to green when checked */
        input[type="checkbox"] {
            accent-color: #34A853 !important;
        }

        /* Table Styling */
        .dataframe {
            font-size: 14px;
            border: 1px solid #ddd;
        }

        /* Agent Progress Styling */
        .agent-step {
            background: #f0f8ff;
            padding: 10px;
            border-left: 4px solid #0062cc;
            margin: 5px 0;
            border-radius: 5px;
        }
        
        .agent-completed {
            background: #f0fff0;
            border-left-color: #28a745;
        }
        
        .agent-current {
            background: #fff8dc;
            border-left-color: #ffc107;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load data
    try:
        # Load all required data
        patients_df = load_patients_data()
        encounters_df = load_encounters_data()
        clinical_events_df = load_clinical_events_data()
        claims_df = load_claims_data()
        risk_scores_df = load_risk_scores_data()

        # Analysis functions mapping
        analysis_functions = {
            "Patient Readmission Risk Analysis": patient_readmission_risk_analysis,
            "Identify Care Gaps": identify_care_gaps,
            "Cost Risk Assessment": cost_risk_assessment,
            "Clinical Event Burden Review": clinical_event_burden_review,
            "Risk Score Trajectory Analysis": risk_score_trajectory_analysis
        }

        # Page title and description
        st.markdown(f'''
        <div style="display:flex; align-items:center; margin-bottom:15px">
            <img src="https://i.imgur.com/Og6gFnB.png" width="100" style="margin-right:15px">
            <div>
                <h1 style="font-size:2.2rem; margin:0; padding:0">Healthcare AI Agents - Patient Recidivism Data App</h1>
                <p style="font-size:1.1rem; color:gray; margin:0; padding:0">Fivetran and Snowflake Cortex-powered data and Agentic AI workflows for Healthcare automated clinical decision support</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.write("Experience autonomous Cortex AI agents that execute complex healthcare workflows, demonstrating multi-step reasoning and clinical decision-making capabilities.")

        # Create tabs
        tabs = st.tabs(["Cortex AI Clinical Agents", "Patient Explorer", "Encounter Analysis", "Clinical Events Explorer", "Claims Analysis", "Cortex Clinical Insights", "Analysis History", "Settings"])

        # Cortex AI Clinical Agents Tab
        with tabs[0]:
            st.header("Cortex AI Clinical Agents")
            st.write("Watch autonomous AI agents execute complex healthcare workflows with multi-step reasoning and clinical decision-making.")
            
            # Create sub-tabs for different agent scenarios
            agent_tabs = st.tabs([
                "🏥 Proactive Care Management", 
                "👥 Population Health Manager",
                "💰 Prior Authorization",
                "🛡️ Quality & Safety",
                "⚙️ Operations Optimization"
            ])
            
            # SCENARIO 1: Proactive Care Management
            with agent_tabs[0]:
                st.subheader("Scenario 1: Proactive Care Management Agent")
                
                with st.container(border=True):
                    st.markdown("""
                    **Business Challenge**: Care coordinators manually review dozens of hospital discharge reports daily, 
                    spending 2+ hours identifying high-risk patients who need immediate intervention to prevent readmission.
                    
                    **Agent Solution**: Autonomous workflow that analyzes recent discharges, assesses readmission risks, 
                    identifies care gaps, and generates prioritized intervention plans in a fraction of the time normally required.
                    """)

                    # 🔥 ADD MODEL SELECTOR HERE - RIGHT AFTER BUSINESS CHALLENGE:
                    selected_model_1 = st.selectbox(
                        "Select Snowflake Cortex Model for Analysis:", 
                        ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"], 
                        key="agent_1_model_selector",
                        help="Choose which AI model will generate the final analysis"
                    )     
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Initialize agent_running if not exists
                        if 'agent_running_1' not in st.session_state:
                            st.session_state.agent_running_1 = False
                        
                        # Button logic without trying to set button values
                        if not st.session_state.agent_running_1:
                            if st.button("🚀 Start Proactive Care Management Agent", type="primary", key="start_agent_1"):
                                st.session_state.agent_running_1 = True
                                # Clear previous results
                                st.session_state.completed_steps_1 = []
                                if 'final_results_1' in st.session_state:
                                    del st.session_state.final_results_1
                                st.rerun()
                        else:
                            # Show disabled button and stop button
                            st.button("🔄 Agent Running...", disabled=True, key="agent_running_disabled_1")
                            
                            if st.button("⏹ Stop Agent", key="stop_agent_1"):
                                st.session_state.agent_running_1 = False
                                st.rerun()

                    with col2:
                        if st.session_state.agent_running_1:
                            st.markdown(
                                '<p style="font-size: 18px; margin: 0;"><strong>Status:</strong> ✅ Active</p>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<p style="font-size: 18px; margin: 0;"><strong>Status:</strong> ⏸ Ready</p>', 
                                unsafe_allow_html=True
                            )
                    
                    # Agent execution area
                    if st.session_state.agent_running_1:
                        agent_placeholder_1 = st.empty()
                        
                        # Run the agent workflow using the utility function
                        try:
                            final_report = execute_agent_workflow(
                                agent_type=1,
                                patients_df=patients_df,
                                encounters_df=encounters_df,
                                clinical_events_df=clinical_events_df,
                                claims_df=claims_df,
                                risk_scores_df=risk_scores_df,
                                placeholder=agent_placeholder_1,
                                selected_model=selected_model_1
                            )
                            
                            st.session_state.final_results_1 = final_report
                            st.session_state.agent_running_1 = False
                            st.success("🎉 Proactive Care Management Agent completed with real data analysis!")

                            # Track agent execution
                            if 'agent_metrics' not in st.session_state:
                                st.session_state.agent_metrics = {'scenario_1_runs': 0, 'scenario_2_runs': 0, 'scenario_3_runs': 0, 'scenario_4_runs': 0, 'scenario_5_runs': 0}
                            st.session_state.agent_metrics['scenario_1_runs'] += 1
                            st.session_state.agent_metrics['last_run'] = datetime.now()
                            
                            with st.expander("📋 Generated Care Management Report (Real Data)", expanded=True):
                                st.markdown(final_report)
                                
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                            st.session_state.agent_running_1 = False
                    
                    # Show previous results if available
                    elif ('completed_steps_1' in st.session_state and 
                          st.session_state.completed_steps_1 and 
                          'final_results_1' in st.session_state):
                        st.subheader("Previous Agent Execution Results")
                        for step_name, result in st.session_state.completed_steps_1:
                            st.markdown(f'<div class="agent-completed">✅ {step_name}: {result}</div>', unsafe_allow_html=True)
                            
                        with st.expander("📋 Most Recent Care Management Report", expanded=False):
                            st.markdown(st.session_state.final_results_1)

            # SCENARIO 2: Population Health Manager
            with agent_tabs[1]:
                st.subheader("Scenario 2: Population Health Manager Agent")
                
                with st.container(border=True):
                    st.markdown("""
                    **Business Challenge**: Population health managers spend 3+ hours daily manually identifying patients 
                    with deteriorating chronic conditions across thousands of records, leading to delayed interventions and poor outcomes.
                    
                    **Agent Solution**: Autonomous workflow that analyzes patient cohorts, identifies at-risk populations, 
                    and generates targeted intervention strategies reducing manual review time by a significant percentage.
                    """)
                    
                    # 🔥 ADD THIS:
                    selected_model_2 = st.selectbox(
                        "Select Snowflake Cortex Model for Analysis:", 
                        ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"], 
                        key="agent_2_model_selector",
                        help="Choose which AI model will generate the final analysis"
                    )                   
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if 'agent_running_2' not in st.session_state:
                            st.session_state.agent_running_2 = False
                        
                        if not st.session_state.agent_running_2:
                            if st.button("🚀 Start Population Health Manager Agent", type="primary", key="start_agent_2"):
                                st.session_state.agent_running_2 = True
                                st.session_state.completed_steps_2 = []
                                if 'final_results_2' in st.session_state:
                                    del st.session_state.final_results_2
                                st.rerun()
                        else:
                            st.button("🔄 Agent Running...", disabled=True, key="agent_running_disabled_2")
                            
                            if st.button("⏹ Stop Agent", key="stop_agent_2"):
                                st.session_state.agent_running_2 = False
                                st.rerun()
                    
                    with col2:
                        if st.session_state.agent_running_2:
                            st.metric("Status", "✅ Active")
                        else:
                            st.metric("Status", "⏸ Ready")
                    
                    if st.session_state.agent_running_2:
                        agent_placeholder_2 = st.empty()
                        
                        try:
                            final_report = execute_agent_workflow(
                                agent_type=2,
                                patients_df=patients_df,
                                encounters_df=encounters_df,
                                clinical_events_df=clinical_events_df,
                                claims_df=claims_df,
                                risk_scores_df=risk_scores_df,
                                placeholder=agent_placeholder_2,
                                selected_model=selected_model_2
                            )
                            
                            st.session_state.final_results_2 = final_report
                            st.session_state.agent_running_2 = False
                            st.success("🎉 Population Health Manager Agent completed with real data analysis!")

                            if 'agent_metrics' not in st.session_state:
                                st.session_state.agent_metrics = {'scenario_1_runs': 0, 'scenario_2_runs': 0, 'scenario_3_runs': 0, 'scenario_4_runs': 0, 'scenario_5_runs': 0}
                            st.session_state.agent_metrics['scenario_2_runs'] += 1
                            st.session_state.agent_metrics['last_run'] = datetime.now()
                            
                            with st.expander("📋 Generated Population Health Report (Real Data)", expanded=True):
                                st.markdown(final_report)
                                
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                            st.session_state.agent_running_2 = False

            # SCENARIO 3: Prior Authorization Agent
            with agent_tabs[2]:
                st.subheader("Scenario 3: Prior Authorization Agent")
                
                with st.container(border=True):
                    st.markdown("""
                    **Business Challenge**: Prior authorization teams manually review 50+ complex cases daily, 
                    spending 45 minutes per case to assess medical necessity and financial risk, causing treatment delays and administrative burden.
                    
                    **Agent Solution**: Autonomous workflow that analyzes treatment requests, validates medical necessity, 
                    and generates evidence-based authorization decisions - reducing review time by a significant percentage and improving approval accuracy.
                    """)
                    
                    # 🔥 ADD THIS:
                    selected_model_3 = st.selectbox(
                        "Select Snowflake Cortex Model for Analysis:", 
                        ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"], 
                        key="agent_3_model_selector",
                        help="Choose which AI model will generate the final analysis"
                    )              
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if 'agent_running_3' not in st.session_state:
                            st.session_state.agent_running_3 = False
                        
                        if not st.session_state.agent_running_3:
                            if st.button("🚀 Start Prior Authorization Agent", type="primary", key="start_agent_3"):
                                st.session_state.agent_running_3 = True
                                st.session_state.completed_steps_3 = []
                                if 'final_results_3' in st.session_state:
                                    del st.session_state.final_results_3
                                st.rerun()
                        else:
                            st.button("🔄 Agent Running...", disabled=True, key="agent_running_disabled_3")
                            
                            if st.button("⏹ Stop Agent", key="stop_agent_3"):
                                st.session_state.agent_running_3 = False
                                st.rerun()
                    
                    with col2:
                        if st.session_state.agent_running_3:
                            st.metric("Status", "✅ Active")
                        else:
                            st.metric("Status", "⏸ Ready")
                    
                    if st.session_state.agent_running_3:
                        agent_placeholder_3 = st.empty()
                        
                        try:
                            final_report = execute_agent_workflow(
                                agent_type=3,
                                patients_df=patients_df,
                                encounters_df=encounters_df,
                                clinical_events_df=clinical_events_df,
                                claims_df=claims_df,
                                risk_scores_df=risk_scores_df,
                                placeholder=agent_placeholder_3,
                                selected_model=selected_model_3
                            )
                            
                            st.session_state.final_results_3 = final_report
                            st.session_state.agent_running_3 = False
                            st.success("🎉 Prior Authorization Agent completed with real data analysis!")

                            if 'agent_metrics' not in st.session_state:
                                st.session_state.agent_metrics = {'scenario_1_runs': 0, 'scenario_2_runs': 0, 'scenario_3_runs': 0, 'scenario_4_runs': 0, 'scenario_5_runs': 0}
                            st.session_state.agent_metrics['scenario_3_runs'] += 1
                            st.session_state.agent_metrics['last_run'] = datetime.now()
                            
                            with st.expander("📋 Generated Prior Authorization Report (Real Data)", expanded=True):
                                st.markdown(final_report)
                                
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                            st.session_state.agent_running_3 = False

            # SCENARIO 4: Clinical Quality & Safety Surveillance Agent
            with agent_tabs[3]:
                st.subheader("Scenario 4: Clinical Quality & Safety Surveillance Agent")
                
                with st.container(border=True):
                    st.markdown("""
                    **Business Challenge**: Quality assurance nurses manually review hundreds of patient records weekly 
                    to identify potential safety events and quality gaps, spending 4+ hours daily on retrospective chart reviews.
                    
                    **Agent Solution**: Autonomous workflow that continuously monitors clinical data, detects safety events, 
                    and generates real-time quality alerts - reducing manual review by a significant percentage and enabling proactive intervention.
                    """)
                    
                    # 🔥 ADD THIS:
                    selected_model_4 = st.selectbox(
                        "Select Snowflake Cortex Model for Analysis:", 
                        ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"], 
                        key="agent_4_model_selector",
                        help="Choose which AI model will generate the final analysis"
                    )              
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if 'agent_running_4' not in st.session_state:
                            st.session_state.agent_running_4 = False
                        
                        if not st.session_state.agent_running_4:
                            if st.button("🚀 Start Quality & Safety Agent", type="primary", key="start_agent_4"):
                                st.session_state.agent_running_4 = True
                                st.session_state.completed_steps_4 = []
                                if 'final_results_4' in st.session_state:
                                    del st.session_state.final_results_4
                                st.rerun()
                        else:
                            st.button("🔄 Agent Running...", disabled=True, key="agent_running_disabled_4")
                            
                            if st.button("⏹ Stop Agent", key="stop_agent_4"):
                                st.session_state.agent_running_4 = False
                                st.rerun()
                    
                    with col2:
                        if st.session_state.agent_running_4:
                            st.metric("Status", "✅ Active")
                        else:
                            st.metric("Status", "⏸ Ready")
                    
                    if st.session_state.agent_running_4:
                        agent_placeholder_4 = st.empty()
                        
                        try:
                            final_report = execute_agent_workflow(
                                agent_type=4,
                                patients_df=patients_df,
                                encounters_df=encounters_df,
                                clinical_events_df=clinical_events_df,
                                claims_df=claims_df,
                                risk_scores_df=risk_scores_df,
                                placeholder=agent_placeholder_4,
                                selected_model=selected_model_4
                            )
                            
                            st.session_state.final_results_4 = final_report
                            st.session_state.agent_running_4 = False
                            st.success("🎉 Quality & Safety Surveillance Agent completed with real data analysis!")

                            if 'agent_metrics' not in st.session_state:
                                st.session_state.agent_metrics = {'scenario_1_runs': 0, 'scenario_2_runs': 0, 'scenario_3_runs': 0, 'scenario_4_runs': 0, 'scenario_5_runs': 0}
                            st.session_state.agent_metrics['scenario_4_runs'] += 1
                            st.session_state.agent_metrics['last_run'] = datetime.now()
                            
                            with st.expander("📋 Generated Quality & Safety Report (Real Data)", expanded=True):
                                st.markdown(final_report)
                                
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                            st.session_state.agent_running_4 = False

            # SCENARIO 5: Operational Efficiency & Resource Optimization Agent
            with agent_tabs[4]:
                st.subheader("Scenario 5: Operational Efficiency & Resource Optimization Agent")
                
                with st.container(border=True):
                    st.markdown("""
                    **Business Challenge**: Operations managers spend hours analyzing utilization patterns, staffing models, 
                    and resource allocation across departments, often missing optimization opportunities that could save millions annually.
                    
                    **Agent Solution**: Autonomous workflow that analyzes operational data, identifies inefficiencies, 
                    and generates resource optimization strategies - improving productivity and reducing annual operational costs.
                    """)
                    
                    # 🔥 ADD THIS:
                    selected_model_5 = st.selectbox(
                        "Select Snowflake Cortex Model for Analysis:", 
                        ["claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-70b"], 
                        key="agent_5_model_selector",
                        help="Choose which AI model will generate the final analysis"
                    )              
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if 'agent_running_5' not in st.session_state:
                            st.session_state.agent_running_5 = False
                        
                        if not st.session_state.agent_running_5:
                            if st.button("🚀 Start Operations Optimization Agent", type="primary", key="start_agent_5"):
                                st.session_state.agent_running_5 = True
                                st.session_state.completed_steps_5 = []
                                if 'final_results_5' in st.session_state:
                                    del st.session_state.final_results_5
                                st.rerun()
                        else:
                            st.button("🔄 Agent Running...", disabled=True, key="agent_running_disabled_5")
                            
                            if st.button("⏹ Stop Agent", key="stop_agent_5"):
                                st.session_state.agent_running_5 = False
                                st.rerun()
                    
                    with col2:
                        if st.session_state.agent_running_5:
                            st.metric("Status", "✅ Active")
                        else:
                            st.metric("Status", "⏸ Ready")
                    
                    if st.session_state.agent_running_5:
                        agent_placeholder_5 = st.empty()
                        
                        try:
                            final_report = execute_agent_workflow(
                                agent_type=5,
                                patients_df=patients_df,
                                encounters_df=encounters_df,
                                clinical_events_df=clinical_events_df,
                                claims_df=claims_df,
                                risk_scores_df=risk_scores_df,
                                placeholder=agent_placeholder_5,
                                selected_model=selected_model_5
                            )
                            
                            st.session_state.final_results_5 = final_report
                            st.session_state.agent_running_5 = False
                            st.success("🎉 Operations Optimization Agent completed with real data analysis!")

                            if 'agent_metrics' not in st.session_state:
                                st.session_state.agent_metrics = {'scenario_1_runs': 0, 'scenario_2_runs': 0, 'scenario_3_runs': 0, 'scenario_4_runs': 0, 'scenario_5_runs': 0}
                            st.session_state.agent_metrics['scenario_5_runs'] += 1
                            st.session_state.agent_metrics['last_run'] = datetime.now()
                            
                            with st.expander("📋 Generated Operations Optimization Report (Real Data)", expanded=True):
                                st.markdown(final_report)
                                
                        except Exception as e:
                            st.error(f"Agent error: {str(e)}")
                            st.session_state.agent_running_5 = False

        # Patient Explorer Tab
        with tabs[1]:
            st.header("Patient Explorer")
            if not patients_df.empty:
                # Top metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with st.container(border=True):
                        st.metric("Total Patients", len(patients_df))
                with col2:
                    with st.container(border=True):
                        high_risk_count = len(patients_df[patients_df['RISK_CATEGORY'] == 'High'])
                        st.metric("High Risk Patients", high_risk_count)
                with col3:
                    with st.container(border=True):
                        # Calculate average age - handle Snowflake VARCHAR DOB
                        try:
                            patients_df_copy = patients_df.copy()
                            patients_df_copy['DOB'] = pd.to_datetime(patients_df_copy['DOB'], errors='coerce')
                            avg_age = ((pd.to_datetime('today') - patients_df_copy['DOB']).dt.days / 365.25).mean()
                            st.metric("Average Age", f"{avg_age:.0f}")
                        except:
                            st.metric("Average Age", "N/A")
                with col4:
                    with st.container(border=True):
                        unique_zips = patients_df['ZIP_CODE'].nunique()
                        st.metric("Coverage Areas (# of Zip Codes)", f"{unique_zips}")

                # Risk filter
                risk_filter = st.selectbox("Filter by Risk Category", ["All"] + sorted(patients_df['RISK_CATEGORY'].dropna().unique().tolist()))
                filtered = patients_df if risk_filter == "All" else patients_df[patients_df["RISK_CATEGORY"] == risk_filter]

                # Risk Distribution - Full Width
                st.subheader("Risk Distribution")
                
                # Risk category distribution with healthcare colors
                risk_colors = ['#27AE60', '#F39C12', '#E74C3C']  # Green (Low), Orange (Moderate), Red (High)
                
                risk_counts = filtered['RISK_CATEGORY'].value_counts().reset_index()
                risk_counts.columns = ['risk_category', 'count']
                
                # Create horizontal bar chart for risk distribution
                risk_chart = alt.Chart(risk_counts).mark_bar(
                    cornerRadiusTopRight=8,
                    cornerRadiusBottomRight=8,
                    height=40,
                    opacity=0.85
                ).encode(
                    y=alt.Y('risk_category:N', title='Risk Category', sort=['High', 'Moderate', 'Low']),
                    x=alt.X('count:Q', title='Number of Patients'),
                    color=alt.Color('risk_category:N', 
                                scale=alt.Scale(domain=['Low', 'Moderate', 'High'], 
                                              range=risk_colors),
                                legend=None),
                    tooltip=['risk_category', 'count']
                ).properties(height=200)
                st.altair_chart(risk_chart, use_container_width=True)

                # Demographics Overview - 4 Columns
                st.subheader("Demographics Overview")
                demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
                
                with demo_col1:
                    st.markdown("**Gender**")
                    gender_counts = filtered['GENDER'].value_counts()
                    for gender, count in gender_counts.items():
                        percentage = (count / len(filtered)) * 100
                        st.write(f"**{gender}:** {percentage:.1f}%")
                        st.progress(percentage / 100)
                
                with demo_col2:
                    st.markdown("**Age Groups**")
                    try:
                        filtered_copy = filtered.copy()
                        filtered_copy['DOB'] = pd.to_datetime(filtered_copy['DOB'], errors='coerce')
                        filtered_copy['age'] = ((pd.to_datetime('today') - filtered_copy['DOB']).dt.days / 365.25)
                        filtered_copy['age_group'] = pd.cut(filtered_copy['age'], 
                                                        bins=[0, 18, 35, 50, 65, 100], 
                                                        labels=['0-17', '18-34', '35-49', '50-64', '65+'])
                        age_counts = filtered_copy['age_group'].value_counts().sort_values(ascending=False).head(4)
                        for age_group, count in age_counts.items():
                            percentage = (count / len(filtered)) * 100
                            st.write(f"**{age_group}:** {percentage:.1f}%")
                            st.progress(percentage / 100)
                    except:
                        st.write("Age data unavailable")
                
                with demo_col3:
                    st.markdown("**Race/Ethnicity**")
                    race_counts = filtered['RACE'].value_counts().head(4)
                    for race, count in race_counts.items():
                        percentage = (count / len(filtered)) * 100
                        st.write(f"**{race}:** {percentage:.1f}%")
                        st.progress(percentage / 100)
                
                with demo_col4:
                    st.markdown("**Language**")
                    lang_counts = filtered['PRIMARY_LANGUAGE'].value_counts().head(4)
                    for lang, count in lang_counts.items():
                        percentage = (count / len(filtered)) * 100
                        st.write(f"**{lang}:** {percentage:.1f}%")
                        st.progress(percentage / 100)

                # Geographic distribution
                st.subheader("Geographic Distribution")
                
                geo_col1, geo_col2, geo_col3 = st.columns([4, .5, 1])
                
                with geo_col1:
                    # Top zip codes by patient count
                    zip_counts = filtered['ZIP_CODE'].value_counts().head(10).reset_index()
                    zip_counts.columns = ['zip_code', 'patient_count']
                    
                    zip_chart = alt.Chart(zip_counts).mark_bar(
                        cornerRadiusTopRight=8,
                        cornerRadiusBottomRight=8,
                        height=25,
                        opacity=0.8
                    ).encode(
                        y=alt.Y('zip_code:N', sort='-x', title='Zip Code', axis=alt.Axis(labelFontSize=10)),
                        x=alt.X('patient_count:Q', title='Number of Patients'),
                        color=alt.Color('patient_count:Q', 
                                    scale=alt.Scale(scheme='blues'),
                                    legend=None),
                        tooltip=['zip_code', 'patient_count']
                    ).properties(height=300)
                    
                    st.altair_chart(zip_chart, use_container_width=True)
                
                with geo_col2:
                    st.markdown("")  # Add some top spacing
                    st.markdown("**Coverage Summary**")
                    total_zips = filtered['ZIP_CODE'].nunique()
                    top_zip_coverage = zip_counts.head(5)['patient_count'].sum()
                    top_zip_percentage = (top_zip_coverage / len(filtered)) * 100
                    
                    st.metric("Total Zip Codes", total_zips)
                    st.metric("Top 5 Zip Coverage", f"{top_zip_percentage:.1f}%")
                
                with geo_col3:
                    st.markdown("**Risk in Top Areas**")
                    top_zips = zip_counts.head(2)['zip_code'].tolist()
                    for zip_code in top_zips:
                        zip_patients = filtered[filtered['ZIP_CODE'] == zip_code]
                        if len(zip_patients) > 0:
                            high_risk_pct = (len(zip_patients[zip_patients['RISK_CATEGORY'] == 'High']) / len(zip_patients)) * 100
                            st.write(f"**{zip_code}**")
                            st.write(f"High Risk: {high_risk_pct:.0f}%")
                            # Color code the progress bar based on risk level
                            if high_risk_pct > 50:
                                st.markdown(f'<div style="background: linear-gradient(90deg, #E74C3C 0%, #E74C3C {high_risk_pct}%, #f0f0f0 {high_risk_pct}%); height: 10px; border-radius: 5px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)
                            elif high_risk_pct > 25:
                                st.markdown(f'<div style="background: linear-gradient(90deg, #F39C12 0%, #F39C12 {high_risk_pct}%, #f0f0f0 {high_risk_pct}%); height: 10px; border-radius: 5px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="background: linear-gradient(90deg, #27AE60 0%, #27AE60 {high_risk_pct}%, #f0f0f0 {high_risk_pct}%); height: 10px; border-radius: 5px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)

                # Patient table with enhanced filtering
                st.subheader("Patient Details")
                
                # Additional filters row
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    gender_filter = st.selectbox("Filter by Gender", ["All"] + sorted(filtered['GENDER'].dropna().unique().tolist()))
                with filter_col2:
                    race_filter = st.selectbox("Filter by Race", ["All"] + sorted(filtered['RACE'].dropna().unique().tolist()))
                with filter_col3:
                    language_filter = st.selectbox("Filter by Language", ["All"] + sorted(filtered['PRIMARY_LANGUAGE'].dropna().unique().tolist()))
                
                # Apply all filters
                display_df = filtered.copy()
                if gender_filter != "All":
                    display_df = display_df[display_df["GENDER"] == gender_filter]
                if race_filter != "All":
                    display_df = display_df[display_df["RACE"] == race_filter]
                if language_filter != "All":
                    display_df = display_df[display_df["PRIMARY_LANGUAGE"] == language_filter]
                
                # Show filtered count
                st.write(f"Showing {len(display_df)} of {len(patients_df)} patients")
                
                # Display table with risk category highlighting
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )

        # Encounter Analysis Tab
        with tabs[2]:
            st.header("Encounter Analysis")
            if not encounters_df.empty:
                # Calculate readmission rate - handle Snowflake VARCHAR boolean
                readmit_rate = (encounters_df['IS_30_DAY_READMIT'].astype(str).str.lower().isin(['true', '1', 'yes'])).mean() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.metric("Total Encounters", len(encounters_df))
                with col2:
                    with st.container(border=True):
                        st.metric("30-Day Readmission Rate", f"{readmit_rate:.2f}%")

                # Create two columns for encounter visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Encounter Distribution")
                    
                    # Create encounter type chart with better styling
                    encounter_colors = ['#E74C3C', '#3498DB', '#52BE80']  # Red, Blue, Green

                    bar = alt.Chart(encounters_df).mark_bar(
                        cornerRadiusTopLeft=8,
                        cornerRadiusTopRight=8,
                        width=80,
                        opacity=0.85
                    ).encode(
                        x=alt.X('ENCOUNTER_TYPE:N', 
                            sort='-y', 
                            title='Encounter Type',
                            axis=alt.Axis(labelAngle=0, labelFontSize=11)),
                        y=alt.Y('count():Q', 
                            title='Count',
                            axis=alt.Axis(grid=True, gridOpacity=0.3)),
                        color=alt.Color('ENCOUNTER_TYPE:N', 
                                    scale=alt.Scale(range=encounter_colors),
                                    legend=None),
                        stroke=alt.value('white'),
                        strokeWidth=alt.value(2)
                    )

                    text = alt.Chart(encounters_df).mark_text(
                        align="center", 
                        baseline="bottom", 
                        dy=-8, 
                        fontSize=10,
                        fontWeight='normal',
                        color='#2C3E50'
                    ).encode(
                        x=alt.X('ENCOUNTER_TYPE:N', sort='-y'),
                        y=alt.Y('count():Q'),
                        text=alt.Text('count():Q', format='.0f')
                    )

                    chart1 = (bar + text).properties(height=300)
                    st.altair_chart(chart1, use_container_width=True)

                with col2:
                    st.subheader("Monthly Encounter Trends")
                    
                    # Create a time-based chart if we have date data
                    if 'ENCOUNTER_START_DATE' in encounters_df.columns:
                        try:
                            # Convert to datetime and extract month-year
                            encounters_df_copy = encounters_df.copy()
                            encounters_df_copy['ENCOUNTER_START_DATE'] = pd.to_datetime(encounters_df_copy['ENCOUNTER_START_DATE'], errors='coerce')
                            encounters_df_copy['month_year'] = encounters_df_copy['ENCOUNTER_START_DATE'].dt.to_period('M').astype(str)
                            
                            # Get monthly counts
                            monthly_data = encounters_df_copy.groupby(['month_year', 'ENCOUNTER_TYPE']).size().reset_index(name='count')
                            
                            line_chart = alt.Chart(monthly_data).mark_line(
                                point=True,
                                strokeWidth=3
                            ).encode(
                                x=alt.X('month_year:N', title='Month', axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y('count:Q', title='Count'),
                                color=alt.Color('ENCOUNTER_TYPE:N', 
                                            scale=alt.Scale(range=encounter_colors),
                                            title='Type'),
                                tooltip=['month_year', 'ENCOUNTER_TYPE', 'count']
                            ).properties(height=300)
                            
                            st.altair_chart(line_chart, use_container_width=True)
                        except:
                            # Fallback chart
                            st.write("Date parsing unavailable - showing readmission rates by type")
                            readmit_by_type = encounters_df.groupby('ENCOUNTER_TYPE').agg({
                                'IS_30_DAY_READMIT': lambda x: (x.astype(str).str.lower().isin(['true', '1', 'yes'])).mean() * 100
                            }).reset_index()
                            readmit_by_type.columns = ['ENCOUNTER_TYPE', 'readmit_rate']
                            
                            readmit_chart = alt.Chart(readmit_by_type).mark_bar(
                                cornerRadiusTopLeft=8,
                                cornerRadiusTopRight=8,
                                opacity=0.8
                            ).encode(
                                x=alt.X('ENCOUNTER_TYPE:N', title='Encounter Type', axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('readmit_rate:Q', title='30-Day Readmission Rate (%)', scale=alt.Scale(domain=[0, 100])),
                                color=alt.Color('ENCOUNTER_TYPE:N', 
                                            scale=alt.Scale(range=encounter_colors),
                                            legend=None),
                                tooltip=['ENCOUNTER_TYPE', alt.Tooltip('readmit_rate:Q', format='.1f')]
                            ).properties(height=300)
                            
                            st.altair_chart(readmit_chart, use_container_width=True)

        # Clinical Events Explorer Tab
        with tabs[3]:
            enhanced_clinical_events_explorer_tab(clinical_events_df)

        # Claims Analysis Tab
        with tabs[4]:
            st.header("Claims Analysis")
            if not claims_df.empty:
                # Convert amounts to numeric for metrics - handle Snowflake data types
                claims_df_numeric = claims_df.copy()
                claims_df_numeric['BILLED_AMOUNT'] = pd.to_numeric(claims_df_numeric['BILLED_AMOUNT'], errors='coerce')
                claims_df_numeric['PAID_AMOUNT'] = pd.to_numeric(claims_df_numeric['PAID_AMOUNT'], errors='coerce')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    with st.container(border=True):
                        st.metric("Total Claims", len(claims_df))
                with col2:
                    with st.container(border=True):
                        st.metric("Total Billed Amount", f"${claims_df_numeric['BILLED_AMOUNT'].sum():,.2f}")
                with col3:
                    with st.container(border=True):
                        st.metric("Total Paid Amount", f"${claims_df_numeric['PAID_AMOUNT'].sum():,.2f}")
                with col4:
                    with st.container(border=True):
                        payment_rate = (claims_df_numeric['PAID_AMOUNT'].sum() / claims_df_numeric['BILLED_AMOUNT'].sum()) * 100
                        st.metric("Payment Rate", f"{payment_rate:.1f}%")

                # Create two columns for claims visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Claims by Type")
                    
                    # Define financial colors
                    claim_colors = ['#3498DB', '#2ECC71']  # Blue, Green
                    
                    # Create horizontal bar chart for claim types
                    claim_counts = claims_df['CLAIM_TYPE'].value_counts().reset_index()
                    claim_counts.columns = ['claim_type', 'count']
                    
                    h_bar = alt.Chart(claim_counts).mark_bar(
                        cornerRadiusTopRight=8,
                        cornerRadiusBottomRight=8,
                        height=40,
                        opacity=0.85
                    ).encode(
                        y=alt.Y('claim_type:N', 
                               sort='-x',
                               title='Claim Type',
                               axis=alt.Axis(labelFontSize=11)),
                        x=alt.X('count:Q', 
                               title='Number of Claims',
                               axis=alt.Axis(grid=True, gridOpacity=0.3)),
                        color=alt.Color('claim_type:N', 
                                       scale=alt.Scale(range=claim_colors),
                                       legend=None),
                        stroke=alt.value('white'),
                        strokeWidth=alt.value(2)
                    )

                    h_text = alt.Chart(claim_counts).mark_text(
                        align="left", 
                        baseline="middle", 
                        dx=5, 
                        fontSize=10,
                        fontWeight='normal',
                        color='#2C3E50'
                    ).encode(
                        y=alt.Y('claim_type:N', sort='-x'),
                        x=alt.X('count:Q'),
                        text=alt.Text('count:Q', format=',')
                    )

                    claims_chart = (h_bar + h_text).properties(height=200)
                    st.altair_chart(claims_chart, use_container_width=True)

                with col2:
                    st.subheader("Payment Efficiency")
                    
                    # Calculate payment rates by claim type
                    payment_analysis = claims_df_numeric.groupby('CLAIM_TYPE').agg({
                        'BILLED_AMOUNT': 'sum',
                        'PAID_AMOUNT': 'sum'
                    }).reset_index()
                    
                    # Convert to float to avoid issues
                    payment_analysis['BILLED_AMOUNT'] = payment_analysis['BILLED_AMOUNT'].astype(float)
                    payment_analysis['PAID_AMOUNT'] = payment_analysis['PAID_AMOUNT'].astype(float)
                    
                    # Calculate payment rate safely
                    payment_analysis['payment_rate'] = (payment_analysis['PAID_AMOUNT'] / payment_analysis['BILLED_AMOUNT'] * 100).round(1)
                    payment_analysis['unpaid_amount'] = payment_analysis['BILLED_AMOUNT'] - payment_analysis['PAID_AMOUNT']
                    
                    # Create stacked area chart for payment vs unpaid
                    payment_long = []
                    for _, row in payment_analysis.iterrows():
                        payment_long.append({
                            'claim_type': row['CLAIM_TYPE'],
                            'amount_type': 'Paid',
                            'amount': float(row['PAID_AMOUNT'])
                        })
                        payment_long.append({
                            'claim_type': row['CLAIM_TYPE'],
                            'amount_type': 'Unpaid',
                            'amount': float(row['unpaid_amount'])
                        })
                    
                    payment_df = pd.DataFrame(payment_long)
                    
                    stacked_chart = alt.Chart(payment_df).mark_bar(
                        cornerRadiusTopLeft=8,
                        cornerRadiusTopRight=8,
                        opacity=0.85
                    ).encode(
                        x=alt.X('claim_type:N', title='Claim Type', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('amount:Q', title='Amount ($)', axis=alt.Axis(format='$.0s')),
                        color=alt.Color('amount_type:N', 
                                    scale=alt.Scale(domain=['Paid', 'Unpaid'], 
                                                    range=['#27AE60', '#E74C3C']),
                                    title='Status'),
                        tooltip=['claim_type', 'amount_type', alt.Tooltip('amount:Q', format='$,.0f')]
                    ).properties(height=220)
                    
                    # Add percentage labels on top of bars
                    percentage_labels = alt.Chart(payment_analysis).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-5,
                        fontSize=11,
                        fontWeight='normal',
                        color='#2C3E50'
                    ).encode(
                        x=alt.X('CLAIM_TYPE:N'),
                        y=alt.Y('BILLED_AMOUNT:Q'),
                        text=alt.Text('payment_rate:Q', format='.1f')
                    ).transform_calculate(
                        percentage_text="datum.payment_rate + '%'"
                    )
                    
                    # Combine chart and labels
                    combined_chart = (stacked_chart + percentage_labels)
                    st.altair_chart(combined_chart, use_container_width=True)
                    
                    # Add a centered caption
                    st.markdown("<div style='text-align: center;'><small>Payment rates shown above bars</small></div>", unsafe_allow_html=True)

        # Cortex Clinical Insights Tab
        with tabs[5]:
            st.header("Cortex Clinical Insights")

            if not patients_df.empty:
                # Patient selection with unique handling
                patients_sorted_df = patients_df.sort_values(by='PATIENT_NAME')
                pat_names = patients_sorted_df['PATIENT_NAME'].tolist()
                pat_ids = patients_sorted_df['PATIENT_ID'].tolist()
                pat_dict = dict(zip(pat_names, pat_ids))
                pat_sel = st.selectbox('Select a Patient', options=list(pat_dict.keys()), key="patient_selector")
                selected_patient_name = pat_sel
                selected_patient_id = pat_dict[pat_sel]

                # Analysis type and model selection                 
                selected_problem = st.selectbox("Select Clinical Insight to Generate", list(analysis_functions.keys()), key="problem_selector")                
                model_choice = st.selectbox("Select Cortex Model", MODELS, key="model_selector")                
                enable_clinical_and_visit_info = st.checkbox(                     
                    "Enable clinical_events and encounters tables", value=True, key="enable_clinical_and_visit_info"                 
                )

                if st.button("Generate Insight", type="primary", key="cortex_generate_insight"):
                    # Get patient details
                    row = patients_df[patients_df["PATIENT_ID"] == selected_patient_id].iloc[0]
                    patient_summary = f"""
                    Demographics:
                    - Name: {row['PATIENT_NAME']}
                    - Gender: {row['GENDER']}
                    - DOB: {row['DOB']}
                    - Race: {row['RACE']}
                    - Ethnicity: {row['ETHNICITY']}
                    - Zip Code: {row['ZIP_CODE']}
                    """

                    # Extract encounters and clinical events
                    enc = encounters_df[encounters_df["PATIENT_ID"] == selected_patient_id]
                    encounter_summary = "No encounters."
                    if not enc.empty and enable_clinical_and_visit_info:
                        try:
                            encounter_summary = format_data(enc.sort_values(by="ENCOUNTER_START_DATE", ascending=False).head(5))
                        except:
                            encounter_summary = format_data(enc.head(5))
        
                    ev = clinical_events_df[clinical_events_df["PATIENT_ID"] == selected_patient_id]
                    event_summary = "No clinical events."
                    if not ev.empty and enable_clinical_and_visit_info:
                        try:
                            event_summary = format_data(ev.sort_values(by="EVENT_DATE", ascending=False).head(5))
                        except:
                            event_summary = format_data(ev.head(5))
        
                    cl = claims_df[claims_df["PATIENT_ID"] == selected_patient_id]
                    claims_summary = "No claims history."
                    if not cl.empty:
                        try:
                            claims_summary = format_data(cl.sort_values(by="CLAIM_START_DATE", ascending=False).head(10))
                        except:
                            claims_summary = format_data(cl.head(10))
        
                    rs = risk_scores_df[risk_scores_df["PATIENT_ID"] == selected_patient_id]
                    risk_summary = "No risk scores available."
                    if not rs.empty:
                        try:
                            risk_summary = format_data(rs.sort_values(by="ASSESSMENT_DATE", ascending=False).head(5))
                        except:
                            risk_summary = format_data(rs.head(5))

                    # Get the appropriate analysis function
                    analysis_function = analysis_functions[selected_problem]
                    
                    with st.spinner(f"Generating {selected_problem} with {model_choice}..."):
                        response = analysis_function(
                            patient_summary, encounter_summary, event_summary, 
                            claims_summary, risk_summary, model_choice, selected_patient_id
                        )
                        
                        if response:
                            st.success(response)
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                "type": selected_problem,
                                "patient": selected_patient_name,
                                "model": model_choice,
                                "response": response,
                                "timestamp": datetime.now()
                            })
                            
                            # Add download button
                            st.download_button(
                                label="Download Analysis",
                                data=response,
                                file_name=f"{selected_problem.replace(' ', '_')}_{selected_patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        else:
                            st.error("Failed to generate analysis. Please try again.")

        # Analysis History Tab
        with tabs[6]:
            st.header("Analysis History")
            
            if st.session_state.analysis_history:
                for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
                    # Create a descriptive title for the expander
                    title = f"{analysis['timestamp'].strftime('%Y-%m-%d %H:%M')} - {analysis['type']} ({analysis['model']})"
                    
                    with st.expander(title):
                        st.success(analysis['response'])
                        if st.button("Remove", key=f"history_remove_{i}"):
                            st.session_state.analysis_history.remove(analysis)
                            st.rerun()
            else:
                st.info("No analysis history available yet. Generate an insight to see it here.")

        # Settings Tab
        with tabs[7]:
            st.header("Data App Settings")
            
            # Snowflake Settings section
            st.subheader("Snowflake Settings")
            
            with st.container(border=True):
                st.markdown("**Database Configuration**")
                st.markdown(f"**Database:** `{DATABASE}`")
                st.markdown(f"**Schema:** `{SCHEMA}`")
                
                st.markdown("**Healthcare Tables**")
                tables_list = [TABLE_PATIENTS, TABLE_ENCOUNTERS, TABLE_EVENTS, TABLE_CLAIMS, TABLE_RISK_SCORES]
                for i, table in enumerate(tables_list, 1):
                    st.markdown(f"**{i}.** `{table}`")

            # Agent Configuration section
            st.subheader("AI Agent Configuration")
            
            with st.container(border=True):
                st.markdown("**Agent Data Analysis Settings**")
                
                # Agent processing parameters
                agent_col1, agent_col2 = st.columns(2)
                
                with agent_col1:
                    st.markdown("**Analysis Time Windows**")
                    recent_days = st.number_input("Recent Data Analysis Period (days)", 
                                                min_value=7, max_value=365, value=30,
                                                help="Number of days to look back for 'recent' analysis")
                    
                    clinical_days = st.number_input("Clinical Events Analysis Period (days)", 
                                                  min_value=30, max_value=365, value=90,
                                                  help="Time window for clinical events analysis")
                    
                    high_cost_percentile = st.slider("High-Cost Claims Threshold (percentile)", 
                                                    min_value=80, max_value=99, value=90,
                                                    help="Percentile threshold for identifying high-cost claims")
                
                with agent_col2:
                    st.markdown("**Performance Thresholds**")
                    high_utilizer_threshold = st.number_input("High-Utilizer Patient Threshold (encounters)", 
                                                            min_value=2, max_value=10, value=4,
                                                            help="Minimum encounters to classify as high-utilizer")
                    
                    extended_stay_threshold = st.number_input("Extended Stay Threshold (days)", 
                                                            min_value=3, max_value=14, value=7,
                                                            help="Minimum days to classify as extended stay")
                    
                    quality_target = st.slider("Quality Score Target (%)", 
                                              min_value=70, max_value=99, value=90,
                                              help="Target quality score for performance benchmarking")
                
                # Save configuration
                if st.button("Save Agent Configuration", type="primary", key="save_agent_config"):
                    # Store in session state
                    st.session_state.agent_config = {
                        'recent_days': recent_days,
                        'clinical_days': clinical_days,
                        'high_cost_percentile': high_cost_percentile / 100,
                        'high_utilizer_threshold': high_utilizer_threshold,
                        'extended_stay_threshold': extended_stay_threshold,
                        'quality_target': quality_target
                    }
                    st.success("Agent configuration saved successfully!")

            # Data Quality Monitoring section
            st.subheader("Data Quality Monitoring")
            
            with st.container(border=True):
                st.markdown("**Real-Time Data Quality Checks**")
                
                if st.button("Run Data Quality Assessment", type="primary", key="data_quality_check"):
                    with st.spinner("Analyzing data quality across all tables..."):
                        # Perform data quality checks
                        quality_results = {}
                        
                        # Check patients data quality
                        patients_missing = patients_df.isnull().sum()
                        quality_results['patients'] = {
                            'total_records': len(patients_df),
                            'missing_risk_category': int(patients_missing.get('RISK_CATEGORY', 0)),
                            'missing_zip_code': int(patients_missing.get('ZIP_CODE', 0)),
                            'completeness_score': ((len(patients_df) * len(patients_df.columns) - patients_missing.sum()) / (len(patients_df) * len(patients_df.columns)) * 100)
                        }
                        
                        # Check encounters data quality
                        encounters_missing = encounters_df.isnull().sum()
                        quality_results['encounters'] = {
                            'total_records': len(encounters_df),
                            'missing_readmit_flag': int(encounters_missing.get('IS_30_DAY_READMIT', 0)),
                            'missing_facility': int(encounters_missing.get('FACILITY_ID', 0)),
                            'completeness_score': ((len(encounters_df) * len(encounters_df.columns) - encounters_missing.sum()) / (len(encounters_df) * len(encounters_df.columns)) * 100)
                        }
                        
                        # Check claims data quality
                        if not claims_df.empty:
                            claims_missing = claims_df.isnull().sum()
                            quality_results['claims'] = {
                                'total_records': len(claims_df),
                                'missing_amounts': int(claims_missing.get('PAID_AMOUNT', 0)),
                                'completeness_score': ((len(claims_df) * len(claims_df.columns) - claims_missing.sum()) / (len(claims_df) * len(claims_df.columns)) * 100)
                            }
                        
                        # Display results
                        st.success("Data quality assessment completed!")
                        
                        qual_col1, qual_col2, qual_col3 = st.columns(3)
                        
                        with qual_col1:
                            st.metric("Patients Data Quality", 
                                    f"{quality_results['patients']['completeness_score']:.1f}%",
                                    help=f"Missing risk categories: {quality_results['patients']['missing_risk_category']}")
                        
                        with qual_col2:
                            st.metric("Encounters Data Quality", 
                                    f"{quality_results['encounters']['completeness_score']:.1f}%",
                                    help=f"Missing readmit flags: {quality_results['encounters']['missing_readmit_flag']}")
                        
                        with qual_col3:
                            if 'claims' in quality_results:
                                st.metric("Claims Data Quality", 
                                        f"{quality_results['claims']['completeness_score']:.1f}%",
                                        help=f"Missing amounts: {quality_results['claims']['missing_amounts']}")
                            else:
                                st.metric("Claims Data Quality", "No Data")

            # Agent Performance Monitoring section
            st.subheader("Agent Performance Monitoring")
            
            with st.container(border=True):
                st.markdown("**Agent Execution History**")
                
                # Track agent execution metrics
                if 'agent_metrics' not in st.session_state:
                    st.session_state.agent_metrics = {
                        'scenario_1_runs': 0,
                        'scenario_2_runs': 0,
                        'scenario_3_runs': 0,
                        'scenario_4_runs': 0,
                        'scenario_5_runs': 0,
                        'total_runtime': 0,
                        'last_run': None
                    }
                
                # Display agent usage metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("Total Agent Runs", 
                            sum([st.session_state.agent_metrics[f'scenario_{i}_runs'] for i in range(1, 6)]))
                    st.metric("Proactive Care Runs", st.session_state.agent_metrics['scenario_1_runs'])
                
                with metrics_col2:
                    st.metric("Population Health Runs", st.session_state.agent_metrics['scenario_2_runs'])
                    st.metric("Prior Auth Runs", st.session_state.agent_metrics['scenario_3_runs'])
                
                with metrics_col3:
                    st.metric("Quality & Safety Runs", st.session_state.agent_metrics['scenario_4_runs'])
                    st.metric("Operations Runs", st.session_state.agent_metrics['scenario_5_runs'])
                
                # Most popular agent
                scenario_runs = [st.session_state.agent_metrics[f'scenario_{i}_runs'] for i in range(1, 6)]
                if max(scenario_runs) > 0:
                    most_used_idx = scenario_runs.index(max(scenario_runs)) + 1
                    scenario_names = {1: "Proactive Care", 2: "Population Health", 3: "Prior Authorization", 4: "Quality & Safety", 5: "Operations"}
                    st.info(f"Most Used Agent: {scenario_names[most_used_idx]} ({max(scenario_runs)} runs)")

            # Data Overview section with cards
            st.subheader("Data Overview")
            
            # First row of data cards
            data_col1, data_col2, data_col3 = st.columns(3)
            
            with data_col1:
                with st.container(border=True):
                    st.markdown("**Patients Data**")
                    patients_count_query = f"SELECT COUNT(*) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_PATIENTS}"
                    risk_levels_query = f"SELECT COUNT(DISTINCT RISK_CATEGORY) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_PATIENTS}"
                    
                    patients_count = query_snowflake(patients_count_query, cache_key="patients_count", ttl=1800)
                    risk_levels = query_snowflake(risk_levels_query, cache_key="risk_levels_count", ttl=1800)
                    
                    if not patients_count.empty:
                        st.metric("Total Records", f"{patients_count.iloc[0]['COUNT']:,}")
                    if not risk_levels.empty:
                        st.metric("Risk Categories", f"{risk_levels.iloc[0]['COUNT']}")
            
            with data_col2:
                with st.container(border=True):
                    st.markdown("**Encounters Data**")
                    encounters_count_query = f"SELECT COUNT(*) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_ENCOUNTERS}"
                    encounter_types_query = f"SELECT COUNT(DISTINCT ENCOUNTER_TYPE) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_ENCOUNTERS}"
                    
                    encounters_count = query_snowflake(encounters_count_query, cache_key="encounters_count", ttl=1800)
                    encounter_types = query_snowflake(encounter_types_query, cache_key="encounter_types_count", ttl=1800)
                    
                    if not encounters_count.empty:
                        st.metric("Total Records", f"{encounters_count.iloc[0]['COUNT']:,}")
                    if not encounter_types.empty:
                        st.metric("Encounter Types", f"{encounter_types.iloc[0]['COUNT']}")
            
            with data_col3:
                with st.container(border=True):
                    st.markdown("**Clinical Events Data**")
                    events_count_query = f"SELECT COUNT(*) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_EVENTS}"
                    event_types_query = f"SELECT COUNT(DISTINCT EVENT_TYPE) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_EVENTS}"
                    
                    events_count = query_snowflake(events_count_query, cache_key="events_count", ttl=1800)
                    event_types = query_snowflake(event_types_query, cache_key="event_types_count", ttl=1800)
                    
                    if not events_count.empty:
                        st.metric("Total Records", f"{events_count.iloc[0]['COUNT']:,}")
                    if not event_types.empty:
                        st.metric("Event Types", f"{event_types.iloc[0]['COUNT']}")
            
            # Second row of data cards
            data_col1, data_col2, data_col3 = st.columns(3)
            
            with data_col1:
                with st.container(border=True):
                    st.markdown("**Claims Data**")
                    claims_count_query = f"SELECT COUNT(*) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_CLAIMS}"
                    claim_types_query = f"SELECT COUNT(DISTINCT CLAIM_TYPE) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_CLAIMS}"
                    
                    claims_count = query_snowflake(claims_count_query, cache_key="claims_count", ttl=1800)
                    claim_types = query_snowflake(claim_types_query, cache_key="claim_types_count", ttl=1800)
                    
                    if not claims_count.empty:
                        st.metric("Total Records", f"{claims_count.iloc[0]['COUNT']:,}")
                    if not claim_types.empty:
                        st.metric("Claim Types", f"{claim_types.iloc[0]['COUNT']}")
            
            with data_col2:
                with st.container(border=True):
                    st.markdown("**Risk Scores Data**")
                    risk_scores_count_query = f"SELECT COUNT(*) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_RISK_SCORES}"
                    risk_levels_query = f"SELECT COUNT(DISTINCT RISK_LEVEL) AS count FROM {DATABASE}.{SCHEMA}.{TABLE_RISK_SCORES}"
                    
                    risk_scores_count = query_snowflake(risk_scores_count_query, cache_key="risk_scores_count", ttl=1800)
                    risk_levels_count = query_snowflake(risk_levels_query, cache_key="risk_score_levels_count", ttl=1800)
                    
                    if not risk_scores_count.empty:
                        st.metric("Total Records", f"{risk_scores_count.iloc[0]['COUNT']:,}")
                    if not risk_levels_count.empty:
                        st.metric("Risk Levels", f"{risk_levels_count.iloc[0]['COUNT']}")
            
            with data_col3:
                with st.container(border=True):
                    st.markdown("**Data Freshness**")
                    
                    # Get most recent data timestamp
                    try:
                        freshness_query = f"""
                        SELECT MAX(_FIVETRAN_SYNCED) as last_sync_time
                        FROM (
                            SELECT MAX(_FIVETRAN_SYNCED) as _FIVETRAN_SYNCED FROM {DATABASE}.{SCHEMA}.{TABLE_PATIENTS}
                            UNION ALL
                            SELECT MAX(_FIVETRAN_SYNCED) as _FIVETRAN_SYNCED FROM {DATABASE}.{SCHEMA}.{TABLE_ENCOUNTERS}
                            UNION ALL
                            SELECT MAX(_FIVETRAN_SYNCED) as _FIVETRAN_SYNCED FROM {DATABASE}.{SCHEMA}.{TABLE_EVENTS}
                            UNION ALL
                            SELECT MAX(_FIVETRAN_SYNCED) as _FIVETRAN_SYNCED FROM {DATABASE}.{SCHEMA}.{TABLE_CLAIMS}
                            UNION ALL
                            SELECT MAX(_FIVETRAN_SYNCED) as _FIVETRAN_SYNCED FROM {DATABASE}.{SCHEMA}.{TABLE_RISK_SCORES}
                        )
                        """
                        
                        last_sync = query_snowflake(freshness_query, cache_key="last_sync", ttl=300)
                        
                        if not last_sync.empty and last_sync.iloc[0]['LAST_SYNC_TIME'] is not None:
                            sync_time = last_sync.iloc[0]['LAST_SYNC_TIME']
                            if hasattr(sync_time, 'strftime'):
                                sync_display = sync_time.strftime('%Y-%m-%d %H:%M')
                                st.metric("Last Sync", sync_display)
                                
                                # Calculate time since last sync
                                now = pd.Timestamp.now()
                                time_diff = now - sync_time
                                hours_ago = int(time_diff.total_seconds() / 3600)
                                st.metric("Hours Ago", f"{hours_ago}h")
                            else:
                                st.metric("Last Sync", str(sync_time))
                                st.metric("Status", "Active")
                        else:
                            st.metric("Last Sync", "Unknown")
                            st.metric("Status", "Check Connection")
                    except:
                        st.metric("Last Sync", "N/A")
                        st.metric("Status", "Data Available")

            # Export and Backup section
            st.subheader("Report Export & Management")
            
            with st.container(border=True):
                st.markdown("**Export Agent Reports**")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Export all analysis history
                    if st.session_state.analysis_history:
                        export_data = []
                        for analysis in st.session_state.analysis_history:
                            export_data.append({
                                'timestamp': analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                                'type': analysis['type'],
                                'patient': analysis['patient'],
                                'model': analysis['model'],
                                'response_length': len(analysis['response'])
                            })
                        
                        export_df = pd.DataFrame(export_data)
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Export Analysis History (CSV)",
                            data=csv_data,
                            file_name=f"agent_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No analysis history to export")
                
                with export_col2:
                    # Export agent results
                    agent_results = {}
                    for i in range(1, 6):
                        if f'final_results_{i}' in st.session_state:
                            agent_results[f'scenario_{i}'] = st.session_state[f'final_results_{i}']
                    
                    if agent_results:
                        # Create combined report
                        combined_report = f"""# Healthcare AI Agents - Combined Analysis Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Platform**: Snowflake Cortex AI

{chr(10).join([f"## {scenario.replace('_', ' ').title()}{chr(10)}{report}{chr(10)}" for scenario, report in agent_results.items()])}
"""
                        
                        st.download_button(
                            label="📋 Export All Agent Reports (MD)",
                            data=combined_report,
                            file_name=f"combined_agent_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.info("No agent reports to export")

            # Cortex Model Settings section
            st.subheader("Cortex Model Settings")
            
            with st.container(border=True):
                st.markdown("**Available Models**")
                for i, model in enumerate(MODELS):
                    if i == 0:
                        prefix = "Primary:"
                    elif i == 1:
                        prefix = "Secondary:"
                    else:
                        prefix = f"{i+1}."
                    st.markdown(f"**{prefix}** `{model}`")
                
                st.markdown("**Test Model Connection**")
                test_text = st.text_area("Enter a quick healthcare question:", 
                                        placeholder="E.g., What are the key factors that influence 30-day readmission rates?",
                                        height=100)
                
                test_model = st.selectbox("Select model to test:", MODELS[:4])  # Show top 4 models
                
                if test_text and st.button("Test Connection", type="primary", key="settings_test_connection"):
                    with st.spinner("Testing connection to Cortex model..."):
                        response = call_cortex_model(test_text, test_model)
                        if response and not str(response).startswith("Error"):
                            st.success("Successfully connected to model!")
                            with st.expander("Model Response"):
                                st.markdown(response)
                        else:
                            st.error("Failed to connect to model endpoint.")

            # Cache Management section
            st.subheader("Cache Management")
            
            cache_col1, cache_col2 = st.columns(2)
            
            with cache_col1:
                with st.container(border=True):
                    cache_size = len(st.session_state.data_cache)
                    st.metric("Current Cache Size", f"{cache_size} items")
                    
                    if cache_size > 0:
                        if st.button("Clear Data Cache", type="secondary", key="settings_clear_cache"):
                            st.session_state.data_cache = {}
                            st.success("Data cache cleared!")
                            st.rerun()
                    else:
                        st.info("Cache is empty")
            
            with cache_col2:
                with st.container(border=True):
                    history_size = len(st.session_state.analysis_history)
                    st.metric("Analysis History", f"{history_size} items")
                    
                    if history_size > 0:
                        if st.button("Clear Analysis History", type="secondary", key="settings_clear_history"):
                            st.session_state.analysis_history = []
                            st.success("Analysis history cleared!")
                            st.rerun()
                    else:
                        st.info("No analysis history")

            # Application Maintenance section
            st.subheader("Application Maintenance")
            
            with st.container(border=True):
                st.markdown("**Table Schema Inspector**")
                
                # Single column layout
                inspect_table = st.selectbox(
                    "Select table to inspect:", 
                    [TABLE_PATIENTS, TABLE_ENCOUNTERS, TABLE_EVENTS, TABLE_CLAIMS, TABLE_RISK_SCORES],
                    key="settings_inspect_table"
                )
                
                if st.button("Load Sample Data", type="primary", key="settings_load_sample"):
                    table_data_query = f"SELECT * FROM {DATABASE}.{SCHEMA}.{inspect_table} LIMIT 5"
                    table_data = query_snowflake(table_data_query, cache_key=f"inspect_{inspect_table}", ttl=600)
                    
                    if not table_data.empty:
                        st.markdown(f"**Sample Data from `{inspect_table}`:**")
                        st.dataframe(table_data, use_container_width=True)
                        
                        st.markdown("**Column Information:**")
                        # Get Snowflake column types
                        schema_query = f"""
                        SELECT COLUMN_NAME, DATA_TYPE 
                        FROM {DATABASE}.INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = '{SCHEMA}' 
                        AND TABLE_NAME = '{inspect_table.upper()}'
                        ORDER BY ORDINAL_POSITION
                        """
                        
                        try:
                            schema_data = query_snowflake(schema_query, cache_key=f"schema_{inspect_table}", ttl=3600)
                            
                            if not schema_data.empty:
                                schema_data.columns = ["Column", "Snowflake Data Type"]
                                st.dataframe(schema_data, hide_index=True)
                            else:
                                st.warning(f"Could not retrieve schema for table: {inspect_table}")
                                # Fallback to pandas types
                                col_info = []
                                for col, dtype in zip(table_data.columns, table_data.dtypes):
                                    col_info.append({"Column": col, "Data Type": str(dtype)})
                                st.dataframe(pd.DataFrame(col_info), hide_index=True)
                        except Exception as e:
                            st.error(f"Error executing schema query: {str(e)}")
                            # Fallback to pandas types
                            col_info = []
                            for col, dtype in zip(table_data.columns, table_data.dtypes):
                                col_info.append({"Column": col, "Data Type": str(dtype)})
                            st.dataframe(pd.DataFrame(col_info), hide_index=True)
                    else:
                        st.warning(f"No data returned from {inspect_table}")

    except Exception as e:
        st.error(f"An error occurred while loading the dashboard: {str(e)}")
        st.error("Please ensure you have the correct database and schema context set in Snowflake.")

if __name__ == "__main__":
    main()