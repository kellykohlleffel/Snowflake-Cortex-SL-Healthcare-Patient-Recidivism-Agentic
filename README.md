# Healthcare AI Agents - Patient Recidivism Data App - Streamlit in Snowflake

## Table of Contents
- [Overview](#overview)
- [AI Clinical Agents - Autonomous Healthcare Workflows](#-ai-clinical-agents---autonomous-healthcare-workflows)
- [Agent Workflow Architecture](#agent-workflow-architecture)
- [Features](#features)
- [Latest Enhancements](#-latest-enhancements)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Data Sources and Schema](#data-sources-and-schema)
- [Troubleshooting](#troubleshooting)

## Overview

The Healthcare AI Agents - Patient Recidivism Data App is a comprehensive analytics platform that transforms healthcare data into actionable insights using **Snowflake Cortex AI** and **autonomous AI agents**. Built as a Streamlit in Snowflake application, this platform demonstrates the power of combining real-time data analysis with AI-powered clinical insights and **autonomous multi-step healthcare workflows**.

**🚀 NEW: Autonomous AI Agents** - The application features five specialized AI agents that execute complex healthcare workflows autonomously, demonstrating agentic AI patterns with multi-step reasoning, clinical decision-making, and seamless "passing of the baton" between analytical processes.

**🔧 Streamlit in Snowflake Architecture** - Single-file application design optimized for Snowflake's native Streamlit environment with integrated Cortex AI capabilities.

The app addresses critical healthcare challenges:
- **Predicting 30-day hospital readmission risk** using AI-driven analysis
- **Identifying care gaps** and coordination failures across patient journeys
- **Analyzing clinical event patterns** and patient care burden with autonomous AI insights
- **Assessing financial risk** and cost impact across patient populations
- **Tracking risk score trajectories** over time with predictive analytics
- **Autonomous clinical workflow execution** through specialized AI agents

## AI Clinical Agents - Autonomous Healthcare Workflows

### Autonomous Healthcare Agent Architecture

The application features **five specialized autonomous AI agents**, each designed as focused clinical analysis tools that behave like **enterprise business applications** with autonomous decision-making capabilities. These agents demonstrate **agentic AI patterns** through:

- **Goal-Oriented Execution**: Each agent has specific clinical objectives and executes multi-step workflows autonomously
- **Environmental Awareness**: Agents understand healthcare domain context and adapt to patient-specific data
- **Multi-Step Reasoning**: Complex analytical workflows that pass data and insights between processing stages
- **Professional Output**: Structured clinical reports ready for healthcare professional use
- **Real-Time Data Integration**: Live analysis of Snowflake healthcare datasets
- **Enhanced Clinical Pattern Recognition**: AI-powered analysis of clinical events with step-by-step progress tracking

### Agent Workflow Orchestration: "Passing of the Baton"

Each agent executes a sophisticated **6-step workflow** that demonstrates autonomous reasoning and seamless process handoffs:

```
Data Ingestion → Analysis → Pattern Recognition → Clinical Reasoning → Intervention Planning → Report Generation
```

#### Agent Execution Pattern:

1. **Data Acquisition Phase** (Steps 1-2)
   - Agent autonomously queries Snowflake for relevant patient data
   - Performs initial data validation and quality assessment
   - **Baton Pass**: Raw data → Structured analysis datasets

2. **Analysis Phase** (Steps 3-4) 
   - Multi-dimensional pattern recognition across clinical domains
   - Risk stratification and cohort identification
   - **Baton Pass**: Structured data → Clinical insights

3. **Clinical Reasoning Phase** (Steps 5-6)
   - AI-powered clinical decision support using Snowflake Cortex
   - Evidence-based recommendation generation
   - **Baton Pass**: Clinical insights → Actionable interventions

### The Five Autonomous Healthcare Agents

#### 🏥 1. Proactive Care Management Agent

**Business Challenge**: Care coordinators manually review dozens of hospital discharge reports daily, spending 2+ hours identifying high-risk patients who need immediate intervention to prevent readmission.

**Agent Solution**: Autonomous workflow that analyzes recent discharges, assesses readmission risks, identifies care gaps, and generates prioritized intervention plans in a fraction of the time normally required.

**Autonomous Workflow**:
- **Step 1**: **Initializing Agent** - Loading historical patient data from specified date range
- **Step 2**: **Analyzing Recent Discharges** - Reviewing discharges from last 30 days of data
- **Step 3**: **Risk Assessment** - Calculating readmission risk scores using historical patterns
- **Step 4**: **Care Gap Analysis** - Analyzing care patterns and follow-up gaps
- **Step 5**: **Clinical Events Review** - Analyzing recent clinical events and generating interventions
- **Step 6**: **Finalizing Report** - Compiling comprehensive care management report

**Impact**: 95% reduction in manual review time, proactive identification of high-risk patients with specific intervention plans.

#### 👥 2. Population Health Manager Agent

**Business Challenge**: Population health managers spend 3+ hours daily manually identifying patients with deteriorating chronic conditions across thousands of records, leading to delayed interventions.

**Agent Solution**: Autonomous workflow that analyzes patient cohorts, identifies at-risk populations, and generates targeted intervention strategies reducing manual review time by a significant percentage.

**Autonomous Workflow**:
- **Step 1**: **Population Segmentation** - Identifying chronic condition cohorts across patient populations
- **Step 2**: **Risk Trajectory Analysis** - Analyzing deteriorating conditions in specified time period
- **Step 3**: **Care Adherence Assessment** - Reviewing care engagement patterns and follow-up compliance
- **Step 4**: **Intervention Prioritization** - Ranking patients by clinical urgency and impact potential
- **Step 5**: **Resource Allocation** - Analyzing care distribution across ZIP codes and encounter types
- **Step 6**: **Outreach Campaign Generation** - Creating targeted intervention strategies for chronic disease management

**Impact**: 85% improvement in chronic disease management efficiency, 28% projected reduction in emergency visits.

#### 💰 3. Prior Authorization Agent

**Business Challenge**: Prior authorization teams manually review 50+ complex cases daily, spending 45 minutes per case to assess medical necessity and financial risk, causing treatment delays and administrative burden.

**Agent Solution**: Autonomous workflow that analyzes treatment requests, validates medical necessity, and generates evidence-based authorization decisions - reducing review time by a significant percentage and improving approval accuracy.

**Autonomous Workflow**:
- **Step 1**: **Claims Pattern Analysis** - Analyzing total claims for high-cost patterns
- **Step 2**: **Medical Necessity Validation** - Validating medical necessity using procedure-diagnosis alignment
- **Step 3**: **Cost-Benefit Modeling** - Analyzing payment patterns and cost optimization opportunities
- **Step 4**: **Denial Risk Assessment** - Evaluating payer-specific approval patterns
- **Step 5**: **Alternative Treatment Analysis** - Analyzing procedure patterns for cost-effective alternatives
- **Step 6**: **Authorization Recommendation** - Generating evidence-based decisions for recent claims

**Impact**: 80% reduction in review time, improved approval accuracy, significant cost optimization opportunities identified.

#### 🛡️ 4. Quality & Safety Surveillance Agent

**Business Challenge**: Quality assurance nurses manually review hundreds of patient records weekly to identify potential safety events, spending 4+ hours daily on retrospective chart reviews.

**Agent Solution**: Autonomous workflow that continuously monitors clinical data, detects safety events, and generates real-time quality alerts - reducing manual review by a significant percentage and enabling proactive intervention.

**Autonomous Workflow**:
- **Step 1**: **Safety Event Detection** - Scanning encounters for safety events and adverse outcomes
- **Step 2**: **Quality Measure Assessment** - Evaluating clinical quality across core condition categories
- **Step 3**: **Provider Performance Analysis** - Analyzing performance variations across facilities
- **Step 4**: **Risk Pattern Recognition** - Detecting systematic risk patterns in patient records
- **Step 5**: **Corrective Action Planning** - Analyzing recent clinical events for improvement opportunities
- **Step 6**: **Regulatory Compliance Check** - Validating compliance across encounter documentation and safety standards

**Impact**: 75% reduction in manual chart review, proactive safety event identification, improved regulatory compliance.

#### ⚙️ 5. Operations Optimization Agent

**Business Challenge**: Operations managers spend hours analyzing utilization patterns and resource allocation, often missing optimization opportunities that could save millions annually.

**Agent Solution**: Autonomous workflow that analyzes operational data, identifies inefficiencies, and generates resource optimization strategies - improving productivity and reducing annual operational costs.

**Autonomous Workflow**:
- **Step 1**: **Capacity Utilization Analysis** - Analyzing encounters across facilities for length of stay patterns
- **Step 2**: **Staffing Pattern Optimization** - Evaluating efficiency across encounter types and facilities
- **Step 3**: **Equipment Utilization Review** - Analyzing clinical resource allocation across recent procedures
- **Step 4**: **Patient Flow Optimization** - Reviewing discharge efficiency and patient flow patterns
- **Step 5**: **Cost Center Performance** - Benchmarking financial efficiency across payers and claim types
- **Step 6**: **Resource Reallocation Recommendations** - Generating optimization strategies across service areas

**Impact**: Identifies millions in annual optimization opportunities, improves operational efficiency, optimizes resource allocation.

### Agentic AI Architecture Characteristics

#### Enterprise-Grade Autonomous Systems

**Zero-Touch Operation**
- **Point-and-Click Interface**: Users select scenario → click "Start Agent" → receive comprehensive report
- **No Prompt Engineering**: Pre-built clinical reasoning frameworks embedded in each agent
- **Consistent Output**: Professional healthcare reports ready for clinical workflows
- **Scalable Architecture**: Easy addition of new agent specialties without system changes

**Multi-Step Clinical Reasoning**
- **Sequential Decision Making**: Each step builds upon previous analysis for deeper insights
- **Contextual Adaptation**: Agents modify approach based on patient-specific data patterns
- **Evidence-Based Logic**: Clinical reasoning follows established healthcare best practices
- **Real-Time Processing**: Live Snowflake data integration for current patient status

**Seamless Workflow Orchestration**
- **Data Flow Management**: Automatic progression from raw data through clinical insights
- **Quality Gates**: Each step validates output before passing to next phase
- **Error Handling**: Graceful degradation with manual override capabilities
- **Audit Trail**: Complete execution tracking for compliance and optimization

## Agent Workflow Architecture

### Multi-Agent Healthcare System

The application implements a **micro-agent ecosystem** where each agent specializes in a specific healthcare domain:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Proactive Care  │    │ Population      │    │ Prior Auth      │
│ Management      │    │ Health Manager  │    │ Agent           │
│ Agent           │    │ Agent           │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐
         │ Quality &       │    │ Operations      │
         │ Safety Agent    │    │ Optimization    │
         │                 │    │ Agent           │
         └─────────────────┘    └─────────────────┘
```

### Agent Execution Pipeline

Each agent follows a standardized **6-phase execution pipeline**:

#### Phase 1: Data Acquisition (Progress: 0-20%)
- **Snowflake Connection**: Establish secure connection to healthcare datasets
- **Data Validation**: Verify data quality and completeness
- **Context Setting**: Load patient populations and date ranges for analysis

#### Phase 2: Pattern Analysis (Progress: 20-40%)
- **Statistical Analysis**: Identify trends, outliers, and significant patterns
- **Cohort Identification**: Segment patients by risk, condition, or utilization
- **Baseline Establishment**: Calculate current performance metrics

#### Phase 3: Clinical Reasoning (Progress: 40-60%)
- **Evidence Integration**: Combine multiple data sources for comprehensive view
- **Risk Assessment**: Apply clinical algorithms and predictive models
- **Gap Identification**: Detect care gaps, safety issues, or inefficiencies

#### Phase 4: AI Enhancement (Progress: 60-80%)
- **Snowflake Cortex Integration**: Leverage large language models for clinical insights
- **Recommendation Engine**: Generate evidence-based intervention strategies
- **Alternative Analysis**: Identify optimization opportunities and alternatives

#### Phase 5: Strategic Planning (Progress: 80-95%)
- **Priority Ranking**: Order interventions by impact and urgency
- **Resource Allocation**: Map interventions to available resources
- **Timeline Development**: Create implementation schedules

#### Phase 6: Report Generation (Progress: 95-100%)
- **Clinical Documentation**: Generate professional healthcare reports
- **Executive Summary**: Provide high-level insights and recommendations
- **Action Planning**: Create specific next steps and accountability measures

### Data Flow Architecture

```
┌─────────────────┐
│   Snowflake     │
│ Healthcare Data │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Agent Data      │
│ Acquisition     │
│ (Single File)   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Multi-Step      │
│ Analysis        │
│ Pipeline        │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Snowflake       │
│ Cortex AI       │
│ Clinical        │
│ Reasoning       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Professional    │
│ Clinical        │
│ Reports         │
└─────────────────┘
```

## Features

### AI Clinical Agents
- **Five Autonomous Healthcare Agents**: Specialized workflows for different clinical domains
- **Multi-Step Reasoning**: 6-phase execution pipeline with seamless process handoffs
- **Real-Time Data Analysis**: Live Snowflake integration with current patient populations
- **Professional Clinical Reports**: Enterprise-ready documentation with actionable insights
- **Zero-Touch Operation**: Point-and-click interface requiring no technical expertise
- **Agent Performance Monitoring**: Execution tracking, metrics, and optimization insights
- **Enhanced Agent Status Management**: Improved start/stop controls with visual status indicators
- **Model Selection**: Choose from multiple Snowflake Cortex models for each agent

### Patient Explorer
- **Interactive Risk Distribution**: Full-width horizontal bar charts with healthcare color coding
- **4-Column Demographics Overview**: Gender, age groups, race/ethnicity, and language preferences with visual progress bars
- **Geographic Analysis**: Patient distribution by zip code with risk assessment in top coverage areas
- **Advanced Filtering**: Multi-dimensional filtering by risk category, gender, race, and language
- **Real-time Updates**: All visualizations update dynamically based on selected filters

### Encounter Analysis
- **Readmission Tracking**: 30-day readmission rate monitoring with trend analysis
- **Encounter Distribution**: Visual breakdown of emergency, inpatient, and outpatient visits
- **Monthly Trends**: Time-series analysis of encounter patterns
- **Interactive Visualizations**: Hover tooltips and detailed breakdowns

### Clinical Events Explorer
- **Enhanced AI Pattern Analysis**: Autonomous clinical intelligence agent for analyzing clinical event data
- **Step-by-Step Progress Tracking**: Real-time progress updates during clinical pattern analysis
- **AI Clinical Insights**: Population-level pattern recognition with concerning trends identification
- **Model Selection**: Choose from multiple Cortex models for clinical reasoning
- **Event Type Analysis**: Distribution of diagnoses, procedures, medications, and lab results
- **Timeline Visualization**: Monthly event trends by type
- **Event Filtering**: Detailed examination by event category
- **Comprehensive Tracking**: Full clinical event history and patterns

### Claims Analysis
- **4-Metric Financial Dashboard**: 
  - Total Claims volume
  - Total Billed Amount
  - Total Paid Amount  
  - **Payment Rate** (collection efficiency percentage)
- **Claims Distribution**: Horizontal bar chart showing professional vs institutional claims
- **Payment Efficiency**: Stacked bar chart with paid/unpaid amounts and payment rates
- **Financial Analytics**: Revenue analysis and cost tracking

### Snowflake Cortex Clinical Insights

#### Gen AI Micro Applications Architecture

The application features **five specialized Gen AI micro applications**, each designed as focused clinical analysis tools that behave like **enterprise business applications**. This architecture is built on four foundational principles:

**Enterprise Application Design Philosophy**

1. **Structured Instructions & Context**
   - Each micro application contains **professionally crafted clinical prompts**
   - Instructions include medical terminology, clinical reasoning frameworks, and healthcare-specific context
   - Built-in **domain expertise** ensures AI understands healthcare workflows and terminology
   - Prompts incorporate **clinical best practices** and evidence-based medicine principles

2. **Standardized Output & Format**
   - **Consistent report structure** across all analysis types
   - Professional healthcare formatting with clear sections and actionable recommendations
   - **Structured data presentation** that integrates seamlessly with clinical workflows
   - Standardized terminology and clinical language that healthcare professionals expect

3. **Rich Data Context as Fuel**
   - **Patient demographics, encounters, clinical events, claims, and risk scores** provide comprehensive context
   - AI responses are **grounded in actual patient data**, not generic medical knowledge
   - **RAG (Retrieval-Augmented Generation)** ensures insights are specific to the individual patient
   - Toggle control demonstrates the **critical value of comprehensive data integration**

4. **Enterprise User Experience**
   - **Zero prompt engineering required** - behaves like any business application
   - **Point-and-click interface** similar to EHR systems and clinical software
   - Users select patient + analysis type, click "Generate" - **that's it**
   - Results delivered in **professional report format** ready for clinical use

**Enterprise-Grade Micro Applications**

Each micro application operates like a specialized clinical tool:

1. **Patient Readmission Risk Analysis**
2. **Care Gap Identification**
3. **Cost Risk Assessment**
4. **Clinical Event Burden Review**
5. **Risk Score Trajectory Analysis**

### Analysis History
- **Complete Audit Trail**: Track all generated clinical insights with timestamps
- **Organized Display**: Expandable entries with patient, model, and analysis type
- **Remove Capability**: Clean up analysis history as needed
- **Searchable Archive**: Review historical insights and decisions

### Settings & Administration
- **Enhanced Agent Configuration**: Configure autonomous agent parameters with detailed thresholds
- **Data Quality Monitoring**: Real-time data quality assessment across all healthcare tables
- **Agent Performance Monitoring**: Comprehensive tracking of autonomous agent execution metrics
- **Report Export & Management**: Export agent reports and analysis history in multiple formats
- **6-Card Data Overview**: 3x2 grid showing comprehensive data statistics
- **Snowflake Integration**: Real-time connection to healthcare datasets
- **Model Management**: Primary/Secondary model configuration and testing
- **Cache Management**: Data cache and analysis history controls
- **Application Maintenance**: Table schema inspector with sample data preview

## Latest Enhancements

### New AI Clinical Pattern Analysis in Clinical Events Explorer

**Clinical Pattern Agent Analysis**
- **Autonomous Clinical Intelligence**: AI agent that automatically analyzes clinical event data
- **Population-Level Pattern Recognition**: Identifies concerning trends and quality issues
- **Step-by-Step Progress Tracking**: Real-time updates during analysis execution
- **Evidence-Based Recommendations**: Generates specific interventions for improving outcomes
- **Model Selection**: Choose from multiple Snowflake Cortex models for analysis
- **Persistent Results**: Previous analysis stored with timestamps for reference

**Business Challenge Addressed**: Clinical analysts spending 3+ hours daily manually reviewing thousands of clinical events across patient populations.

**Agent Solution**: Autonomous clinical intelligence that identifies concerning trends, detects anomalies, and generates evidence-based recommendations.

### Enhanced Agent Execution with Visual Status Management

**Improved Agent Controls**
- **Visual Status Indicators**: Clear "✅ Active" and "⏸ Ready" status displays
- **Start/Stop Agent Management**: Reliable agent execution control with proper state management
- **Previous Results Display**: Shows completed agent steps and final reports from previous runs
- **Model Selection per Agent**: Choose specific Cortex models for each agent type
- **Enhanced Error Handling**: Better error recovery and user feedback

### Advanced Settings and Monitoring Dashboard

**Agent Configuration Section**
- **Analysis Time Windows**: Configure recent data periods and clinical event analysis windows
- **Performance Thresholds**: Set high-utilizer criteria, extended stay definitions, and quality targets
- **Configuration Persistence**: Save agent settings for consistent performance

**Data Quality Monitoring**
- **Real-Time Quality Checks**: Analyze data completeness across all healthcare tables
- **Quality Metrics**: Track missing data, completeness scores, and data integrity
- **Visual Quality Dashboard**: Color-coded metrics for quick assessment

**Agent Performance Monitoring**
- **Execution History**: Track total agent runs across all five scenarios
- **Usage Analytics**: Monitor most popular agents and execution patterns
- **Performance Metrics**: Track agent completion rates and execution times

**Report Export & Management**
- **Analysis History Export**: Export clinical insights to CSV format
- **Agent Reports Export**: Download combined agent reports in Markdown format
- **Comprehensive Documentation**: Professional reports ready for clinical workflows

### Application Maintenance and Schema Inspection

**Table Schema Inspector**
- **Sample Data Preview**: View sample records from any healthcare table
- **Column Information**: Display Snowflake data types and schema details
- **Data Validation**: Verify table structure and data availability
- **Troubleshooting Support**: Debug data issues and connectivity problems

## Architecture

The application follows a modern data architecture pattern optimized for healthcare analytics with **autonomous AI agent integration** in **Streamlit in Snowflake**:

| Step | Component | Description |
|:-----|:----------|:------------|
| 1    | **Source Systems** | Electronic Health Records (EHR), Practice Management, Hospital Information Systems |
| 2    | **Fivetran Data Movement** | Automated, real-time data replication from source databases to Snowflake |
| 3    | **Snowflake Data Cloud** | Cloud data warehouse storing healthcare data in `HOL_DATABASE.PATIENT_RECIDIVISM_HEALTHCARE` |
| 4    | **Streamlit in Snowflake** | Native Streamlit application with real-time data visualization and analysis |
| 5    | **🤖 AI Agent Orchestration** | Autonomous multi-step healthcare workflows with clinical reasoning |
| 6    | **Snowflake Cortex Integration** | Native Large Language Models (LLMs) providing AI-powered clinical insights |

**Enhanced Data Flow**: Source → Fivetran → Snowflake → **AI Agents** → Streamlit in Snowflake → Cortex AI → Clinical Insights

### AI Agent Architecture Integration

```
┌─────────────────┐
│ Healthcare      │
│ Source Systems  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Fivetran        │
│ Data Pipeline   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Snowflake       │
│ Data Cloud      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│ Single File     │◄──►│ Snowflake       │
│ Agent App       │    │ Cortex AI       │
└─────────┬───────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐
│ Streamlit in    │
│ Snowflake UI    │
└─────────────────┘
```

## Prerequisites

### Required Snowflake Features
- **Snowflake Account** with Streamlit in Snowflake enabled
- **Snowflake Cortex** access with LLM function permissions
- **Database and Schema** with healthcare dataset access
- **Compute Warehouse** with appropriate size for agent execution

### Database Requirements
- **HOL_DATABASE** database access
- **PATIENT_RECIDIVISM_HEALTHCARE** schema access
- **Healthcare Tables**: patients, encounters, clinical_events, claims, risk_scores

### Snowflake Cortex Models
- Access to Cortex LLM functions
- Model availability in your Snowflake region:
  - openai-gpt-oss-20b
  - openai-gpt-4.1
  - claude-4-sonnet
  - claude-3-7-sonnet 
  - claude-3-5-sonnet
  - llama3.1-70b
  - deepseek-r1
  - snowflake-arctic
  - plus many more

## Setup and Installation

### 1. Snowflake Environment Setup

#### Database Context Configuration
Ensure your Snowflake account has the required database and schema:
```sql
USE DATABASE HOL_DATABASE;
USE SCHEMA PATIENT_RECIDIVISM_HEALTHCARE;
```

#### Verify Table Access
Confirm access to required healthcare tables:
```sql
-- Verify table existence and access
SELECT COUNT(*) FROM patients;
SELECT COUNT(*) FROM encounters;
SELECT COUNT(*) FROM clinical_events;
SELECT COUNT(*) FROM claims;
SELECT COUNT(*) FROM risk_scores;
```

#### Enable Cortex Functions
Verify Cortex AI access:
```sql
-- Test Cortex LLM function
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'claude-3-5-sonnet',
    'What are the key factors in healthcare analytics?'
) as test_response;
```

### 2. Streamlit in Snowflake Application Setup

#### Create New Streamlit App
1. Navigate to **Snowsight** → **Streamlit**
2. Click **"+ Streamlit App"**
3. Configure:
   - **Name**: `Healthcare AI Agents - Patient Recidivism`
   - **Warehouse**: Select appropriate compute warehouse
   - **App location**: Choose database and schema for app storage

#### Application Configuration
Set the database context in your Streamlit app:
1. **Database**: `HOL_DATABASE`
2. **Schema**: `PATIENT_RECIDIVISM_HEALTHCARE`
3. **Warehouse**: Assign compute warehouse with sufficient resources for AI agents

### 3. Application Deployment

#### Upload Application Code
1. **Copy the complete `app.py` code** from this repository
2. **Paste into Streamlit editor** in Snowflake
3. **Save and deploy** the application

#### Verify Dependencies
The application uses only Snowflake-native libraries:
- `streamlit` (pre-installed in Snowflake)
- `pandas` (pre-installed)
- `altair` (pre-installed) 
- `snowflake.snowpark.context` (native Snowflake)

No additional package installations required!

## Running the Application

### Development and Testing
1. **Open Streamlit in Snowflake**
2. **Navigate to your application**
3. **Set database context** to `HOL_DATABASE.PATIENT_RECIDIVISM_HEALTHCARE`
4. **Run the application**

### AI Agent Execution
1. **Navigate to "Cortex AI Clinical Agents" tab**
2. **Select desired healthcare scenario**:
   - 🏥 Proactive Care Management
   - 👥 Population Health Manager  
   - 💰 Prior Authorization
   - 🛡️ Quality & Safety
   - ⚙️ Operations Optimization
3. **Choose Snowflake Cortex Model** for analysis
4. **Click "🚀 Start [Agent Name] Agent"**
5. **Watch autonomous execution** with real-time progress updates:
   ```
   🔄 Agent Running...
   ⏹ Stop Agent
   Status: ✅ Active
   
   Step Progress: [▓▓▓▓▓▓▓░░░] 70%
   Current Step: Risk Pattern Recognition
   ```
6. **Review generated clinical report** with actionable insights

### Clinical Events Pattern Analysis
1. **Navigate to "Clinical Events Explorer" tab**
2. **View event type distribution and timeline**
3. **Click "🔍 Analyze Patterns"** for AI-powered insights
4. **Select Snowflake Cortex Model** for analysis
5. **Watch step-by-step progress** with real-time updates
6. **Review AI Clinical Insights** with evidence-based recommendations

### Production Usage
- **Share application** with healthcare team members
- **Configure permissions** for appropriate access levels
- **Monitor usage** through Snowflake account administration
- **Scale compute** based on concurrent user requirements

## Data Sources and Schema

### Healthcare Data Sources

The application integrates with various healthcare systems through Fivetran connectors for **real-time AI agent analysis**:

#### Electronic Health Records (EHR)
- **Epic MyChart, Cerner, Allscripts**
- Patient demographics, encounters, clinical events
- Real-time data synchronization for agent workflows

#### Practice Management Systems
- **Athenahealth, NextGen Healthcare, eClinicalWorks**
- Scheduling, billing, patient management
- Automated data extraction for operational optimization

#### Hospital Information Systems
- **Meditech, Epic Hospital, Cerner**
- Inpatient encounters, admission/discharge data
- Readmission tracking and outcomes for care management agents

#### Laboratory Information Systems
- **LabCorp, Quest Diagnostics, Hospital Labs**
- Lab results, diagnostic codes
- Clinical event tracking for quality and safety agents

#### Revenue Cycle Management
- **Epic Revenue Cycle, Cerner**
- Claims processing, billing data
- Financial analytics and reporting for prior authorization agents

### Snowflake Schema

#### Database Configuration
- **Database**: `HOL_DATABASE`
- **Schema**: `PATIENT_RECIDIVISM_HEALTHCARE`

#### Patients Table (`patients`)
```sql
PATIENT_ID         VARCHAR    -- Unique patient identifier
PATIENT_NAME       VARCHAR    -- Patient full name
DOB                VARCHAR    -- Date of birth
GENDER             VARCHAR    -- Patient gender
ADDRESS            VARCHAR    -- Full address
ZIP_CODE           VARCHAR    -- ZIP code for geographic analysis
RACE               VARCHAR    -- Patient race
ETHNICITY          VARCHAR    -- Patient ethnicity
PRIMARY_LANGUAGE   VARCHAR    -- Preferred language
RISK_CATEGORY      VARCHAR    -- Risk stratification (Low, Moderate, High)
_FIVETRAN_SYNCED   TIMESTAMP_TZ -- Last sync timestamp
_FIVETRAN_DELETED  BOOLEAN    -- Deletion flag
```

#### Risk Scores Table (`risk_scores`)
```sql
RISK_ID            VARCHAR    -- Unique risk assessment identifier
PATIENT_ID         VARCHAR    -- Foreign key to patients
PATIENT_NAME       VARCHAR    -- Patient name
ASSESSMENT_DATE    VARCHAR    -- Date of risk assessment
RISK_SCORE         NUMBER     -- Numerical risk score (0-100)
RISK_LEVEL         VARCHAR    -- Low, Moderate, High
KEY_RISK_FACTORS   VARCHAR    -- Contributing risk factors
_FIVETRAN_SYNCED   TIMESTAMP_TZ -- Last sync timestamp
_FIVETRAN_DELETED  BOOLEAN    -- Deletion flag
```

#### Encounters Table (`encounters`)
```sql
ENCOUNTER_ID              VARCHAR    -- Unique encounter identifier
PATIENT_ID                VARCHAR    -- Foreign key to patients
PATIENT_NAME              VARCHAR    -- Patient name
ENCOUNTER_START_DATE      VARCHAR    -- Encounter start date
ENCOUNTER_END_DATE        VARCHAR    -- Encounter end date
ENCOUNTER_TYPE            VARCHAR    -- Emergency, Inpatient, Outpatient
FACILITY_ID               VARCHAR    -- Healthcare facility identifier
ADMITTING_DIAGNOSIS_CODE  VARCHAR    -- Primary admission diagnosis
DISCHARGE_DIAGNOSIS_CODE  VARCHAR    -- Primary discharge diagnosis
DISCHARGE_DISPOSITION     VARCHAR    -- Discharge destination
IS_30_DAY_READMIT         VARCHAR    -- 30-day readmission flag
PRIOR_ENCOUNTER_ID        VARCHAR    -- Previous encounter reference
_FIVETRAN_SYNCED          TIMESTAMP_TZ -- Last sync timestamp
_FIVETRAN_DELETED         BOOLEAN    -- Deletion flag
```

#### Clinical Events Table (`clinical_events`)
```sql
EVENT_ID         VARCHAR    -- Unique event identifier
PATIENT_ID       VARCHAR    -- Foreign key to patients
PATIENT_NAME     VARCHAR    -- Patient name
ENCOUNTER_ID     VARCHAR    -- Related encounter
EVENT_DATE       VARCHAR    -- Date of clinical event
EVENT_TYPE       VARCHAR    -- Diagnosis, Procedure, Medication, Lab
CODE_TYPE        VARCHAR    -- ICD-10, CPT, LOINC, NDC
CODE_VALUE       VARCHAR    -- Medical code value
DESCRIPTION      VARCHAR    -- Human-readable description
_FIVETRAN_SYNCED TIMESTAMP_TZ -- Last sync timestamp
_FIVETRAN_DELETED BOOLEAN   -- Deletion flag
```

#### Claims Table (`claims`)
```sql
CLAIM_ID           VARCHAR    -- Unique claim identifier
PATIENT_ID         VARCHAR    -- Foreign key to patients
PATIENT_NAME       VARCHAR    -- Patient name
ENCOUNTER_ID       VARCHAR    -- Related encounter
CLAIM_START_DATE   VARCHAR    -- Claim service start date
CLAIM_END_DATE     VARCHAR    -- Claim service end date
PAYER_ID           VARCHAR    -- Insurance payer identifier
CLAIM_TYPE         VARCHAR    -- Professional, Institutional
PROCEDURE_CODE     NUMBER     -- Primary procedure code
DIAGNOSIS_CODE     VARCHAR    -- Primary diagnosis code
BILLED_AMOUNT      NUMBER     -- Amount billed to payer
PAID_AMOUNT        NUMBER     -- Amount paid by payer
_FIVETRAN_SYNCED   TIMESTAMP_TZ -- Last sync timestamp
_FIVETRAN_DELETED BOOLEAN   -- Deletion flag
```

## Troubleshooting

### AI Agent Issues

#### Agent Execution Failures
```bash
Error: Agent workflow interrupted or failed to complete
```

**Solutions:**
1. **Check Snowflake Connectivity**: Verify database context and query permissions
2. **Validate Cortex Access**: Ensure Cortex LLM functions are accessible in your region
3. **Review Data Quality**: Use Settings → Data Quality Assessment
4. **Monitor Warehouse Usage**: Check for compute resource limitations
5. **Restart Agent**: Use "Stop Agent" button and restart execution

#### Agent Performance Optimization
- **Warehouse Sizing**: Ensure adequate compute resources for large datasets
- **Query Optimization**: Complex joins may need optimization for faster processing
- **Model Selection**: Switch between available Cortex models based on performance
- **Concurrent Execution**: Avoid running multiple agents simultaneously on large datasets

#### Agent Report Generation Issues
```bash
Error: Failed to generate clinical report or incomplete analysis
```

**Solutions:**
1. **Verify Data Completeness**: Ensure all required tables have sufficient data
2. **Check Date Ranges**: Validate that analysis periods contain relevant data
3. **Review Agent Configuration**: Adjust thresholds in Settings → Agent Configuration
4. **Manual Retry**: Stop and restart agent execution
5. **Export Partial Results**: Use analysis history to recover partial insights

### Clinical Events Explorer Issues

#### Clinical Pattern Analysis Failures
```bash
Error: Clinical pattern analysis could not be completed
```

**Solutions:**
1. **Verify Clinical Events Data**: Ensure clinical_events table has sufficient data
2. **Check Model Selection**: Try different Cortex models for analysis
3. **Review Event Type Distribution**: Verify event types are properly populated
4. **Clear Previous Analysis**: Remove cached analysis results
5. **Restart Pattern Analysis**: Click "🔍 Analyze Patterns" again

#### Missing Clinical Insights
- **Check Previous Analysis**: Review stored clinical analysis in session state
- **Verify Model Access**: Ensure selected Cortex model is available
- **Review Progress Updates**: Monitor step-by-step progress for failures
- **Check Event Data Quality**: Verify EVENT_TYPE and CODE_VALUE columns are populated

### Snowflake Connectivity Issues

#### Database Context Problems
```bash
Error: Object does not exist or not authorized
```

**Solutions:**
1. Verify database context is set to `HOL_DATABASE.PATIENT_RECIDIVISM_HEALTHCARE`
2. Check permissions for schema and table access
3. Confirm role has appropriate privileges for Cortex functions
4. Test with: `SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()`

#### Cortex Function Access Denied
```bash
Error: Function SNOWFLAKE.CORTEX.COMPLETE not found
```

**Solutions:**
1. Verify **Cortex** is enabled in your Snowflake account
2. Check **regional availability** of Cortex LLM functions
3. Confirm **role permissions** for Cortex access
4. Test Cortex availability with simple function call
5. Contact Snowflake support for Cortex enablement

### Application Performance

#### Slow Query Performance
- **Optimize warehouse size** for better query performance
- **Add appropriate indexes** on frequently queried columns for agent workflows
- **Implement data clustering** on date columns for temporal analysis
- **Use query result caching** (automatically handled by Snowflake)
- **Optimize agent data sampling** for large patient populations

#### Memory Issues
- **Scale warehouse size** for large datasets in agent analysis
- **Implement data pagination** for large result sets
- **Clear cache periodically** using Settings tab
- **Monitor query result sizes** especially for population-level agents
- **Run agents sequentially** rather than concurrently for large datasets

#### Cortex AI Timeouts
```bash
Error: Request timeout calling Snowflake Cortex
```

**Solutions:**
1. **Reduce prompt complexity** for faster response in agent workflows
2. **Check model availability** in your Snowflake region
3. **Verify Cortex quotas** haven't been exceeded
4. **Switch to backup model** (e.g., claude-3-5-sonnet) in agent configuration
5. **Implement retry logic** for transient failures

### Data Quality Issues

#### Missing Data in Visualizations and Agent Analysis
- **Check Fivetran sync status** in Settings tab
- **Verify data types** using Schema Inspector
- **Review filter conditions** that might exclude data
- **Check for null values** in key columns
- **Run Data Quality Assessment** in Settings → Data Quality Monitoring

#### Incorrect Agent Calculations
- **Verify column data types** (especially for financial calculations in Prior Auth agent)
- **Check for missing or invalid date formats** affecting temporal analysis
- **Review aggregation logic** in Snowflake queries
- **Test with sample data** using Schema Inspector
- **Validate agent configuration parameters** in Settings

### Streamlit in Snowflake Issues

#### App Loading Errors
```bash
Error: Unable to load Streamlit application
```

**Solutions:**
1. **Check warehouse status** and ensure it's running
2. **Verify app permissions** and role access
3. **Restart warehouse** if experiencing issues
4. **Check app file size** and complexity
5. **Review Snowflake account limits** for Streamlit apps

#### Session Management Issues
```bash
Error: get_active_session() failed
```

**Solutions:**
1. **Verify Streamlit in Snowflake** is properly enabled
2. **Check warehouse assignment** for the application
3. **Restart Streamlit app** from Snowsight interface
4. **Verify role permissions** for session access
5. **Contact Snowflake support** for session issues

### Enhanced Settings Features Troubleshooting

#### Agent Configuration Issues
```bash
Error: Agent configuration save failed
```

**Solutions:**
1. **Check parameter ranges**: Ensure values are within valid ranges
2. **Verify session state**: Clear browser cache if configuration isn't persisting
3. **Review warehouse permissions**: Ensure adequate resources for configuration
4. **Restart application**: Refresh Streamlit app if configuration issues persist

#### Data Quality Assessment Failures
```bash
Error: Data quality assessment could not complete
```

**Solutions:**
1. **Check table permissions**: Verify access to all healthcare tables
2. **Monitor warehouse resources**: Ensure adequate compute for analysis
3. **Review table structure**: Use Schema Inspector to verify table schemas
4. **Check data volume**: Large tables may require warehouse scaling

#### Schema Inspector Issues
```bash
Error: Could not retrieve schema information
```

**Solutions:**
1. **Verify INFORMATION_SCHEMA access**: Check permissions for schema queries
2. **Check table names**: Ensure table names match exactly (case-sensitive)
3. **Review database context**: Confirm correct database and schema selection
4. **Try alternative tables**: Test with different healthcare tables

### Performance Monitoring

#### Query Performance
- Monitor execution times in Snowflake Query History
- Use query profiling for optimization
- Implement query result caching (automatically handled)
- Consider materialized views for complex aggregations used by agents

#### Application Metrics
- Monitor Streamlit performance through Snowflake administration
- Track warehouse usage during agent execution
- Monitor Cortex AI response times for agent workflows
- **Track agent execution metrics** in Settings → Agent Performance Monitoring

#### AI Agent Performance
- **Execution Time Monitoring**: Track agent completion times across scenarios
- **Success Rate Analysis**: Monitor agent completion vs. failure rates
- **Resource Utilization**: Track warehouse usage during agent execution
- **Cost Optimization**: Monitor Cortex AI usage and costs
- **Report Quality Assessment**: Review generated report completeness and accuracy

## Advanced Configuration

### Custom AI Agent Configuration
The application supports extensive agent customization:

1. **Analysis Time Windows**: Configure recent data periods and clinical event windows
2. **Performance Thresholds**: Set high-utilizer criteria and extended stay definitions
3. **Quality Targets**: Define benchmarks for agent performance assessment
4. **Cost Thresholds**: Configure high-cost percentiles for financial analysis

### Custom Cortex Model Configuration
The application supports multiple Snowflake Cortex models for agent workflows:

1. **Primary Models**: claude-4-sonnet, claude-3-7-sonnet, claude-3-5-sonnet
2. **Alternative Models**: llama3.1-70b, deepseek-r1, snowflake-arctic
3. **Test connectivity** using Settings tab
4. **Configure primary/secondary models** for agent fallback scenarios

### Data Refresh Configuration
- **Real-time**: Fivetran provides continuous data synchronization for agent analysis
- **Auto-refresh**: Snowflake automatically handles query result caching
- **Manual Refresh**: Clear cache in Settings tab
- **Agent Data Validation**: Automatic data quality checks before agent execution

### Security Best Practices
- **Use appropriate roles** with least privilege access for both data and Cortex
- **Monitor access logs** in Snowflake for agent activity
- **Enable query logging** for audit trails of agent executions
- **Implement data masking** for sensitive patient information in agent reports
- **Regular permission reviews** for Streamlit app access

## Performance Considerations

- **AI Agent Analysis**: Typically 60-120 seconds per autonomous workflow using live data
- **Query Caching**: Results automatically cached by Snowflake with intelligent TTL
- **Connection Management**: Native Snowflake session handling for optimal performance
- **Data Processing**: Warehouse auto-scaling for agent performance optimization
- **Agent Orchestration**: Sequential execution prevents resource conflicts
- **Cortex Rate Limiting**: Built-in retry logic for model quota management
- **Clinical Pattern Analysis**: 30-60 seconds for population-level clinical insights

## Agent Workflow Examples

### Example 1: Proactive Care Management Agent Execution

```
🏥 Proactive Care Management Agent Starting...

Step 1/6: Data Acquisition (15% complete)
├─ Querying recent hospital discharges from Snowflake
├─ Found 12 patients discharged in past 48 hours
└─ ✅ Data acquisition complete

Step 2/6: Risk Assessment (35% complete)  
├─ Analyzing readmission risk using historical patterns
├─ Calculated risk scores for 12 patients
└─ ✅ Risk stratification complete

Step 3/6: Pattern Recognition (55% complete)
├─ Identified 4 high-risk patients (≥70% readmission risk)
├─ Prioritized by clinical urgency and intervention potential  
└─ ✅ High-risk patient prioritization complete

Step 4/6: Care Gap Analysis (75% complete)
├─ Analyzing care patterns and follow-up gaps
├─ Found 8 care gaps across high-risk patients
└─ ✅ Care gap identification complete

Step 5/6: Clinical Reasoning (90% complete)
├─ Generating AI-powered intervention recommendations
├─ Created personalized intervention plans using Cortex AI
└─ ✅ Clinical recommendations generated

Step 6/6: Report Generation (100% complete)
├─ Compiling comprehensive care management report
├─ Generated executive summary with actionable insights
└─ ✅ Care coordinator handoff report ready

🎉 Agent completed successfully! 
📋 Professional clinical report generated with 4 priority interventions
⏱️ Process time: 90 seconds (vs. 2+ hours manual review)
```

### Example 2: Clinical Events Pattern Analysis

```
🔬 Clinical Pattern Agent Analysis Starting...

Step 1/4: Data Analysis Initialization (25% complete)
├─ 🔍 Loading and processing clinical event data...
├─ Analyzed 15,847 clinical events across 4 event types
└─ ✅ Data Analysis Initialization complete

Step 2/4: Population Pattern Recognition (50% complete)
├─ 📊 Identifying population-level clinical patterns...
├─ Processed 4 event types with 15,847 total events
└─ ✅ Population Pattern Recognition complete

Step 3/4: Clinical AI Reasoning (75% complete)
├─ ⚙️ Processing clinical data through advanced AI reasoning...
├─ Sending comprehensive analysis to claude-4-sonnet
└─ ✅ Clinical AI Reasoning complete

Step 4/4: Analysis Complete (100% complete)
├─ ✅ Clinical insights generation completed successfully
├─ AI-powered clinical analysis report ready for review
└─ ✅ Analysis Complete

🎉 Clinical Pattern Analysis completed!
📋 AI Clinical Insights generated with concerning trends identified
💡 Evidence-based recommendations for improving patient outcomes
```

## Snowflake-Specific Features

### Native Integration Benefits
- **Single Platform**: Data storage, processing, and AI in one environment
- **Auto-scaling**: Warehouse resources scale automatically for agent workloads
- **Security**: Built-in data governance and access controls
- **Performance**: Optimized query execution for healthcare analytics

### Cortex AI Advantages
- **Native LLMs**: No external API calls required
- **Data Privacy**: All processing stays within Snowflake environment
- **Model Variety**: Multiple LLM options for different use cases
- **Cost Efficiency**: Pay-per-use model for AI functions

### Streamlit in Snowflake Benefits
- **No Infrastructure**: Zero setup required for application hosting
- **Automatic Scaling**: Handles concurrent users seamlessly
- **Version Control**: Built-in versioning for application updates
- **Collaboration**: Easy sharing within Snowflake organization

## Contributing

1. **Fork** the repository or create feature branch in Snowflake
2. **Develop new agents** following the 6-step workflow pattern
3. **Test thoroughly** using provided test scripts and real healthcare data
4. **Update single app.py file** maintaining Streamlit in Snowflake compatibility
5. **Submit updates** with detailed description of agent capabilities

### Adding New AI Agents

To add new autonomous healthcare agents:

1. **Create Agent Function**: Follow the pattern in existing agent implementations
2. **Implement 6-Step Workflow**: Data acquisition → Analysis → Recognition → Reasoning → Planning → Report
3. **Add UI Integration**: Include agent in the main application tabs
4. **Configure Data Sources**: Ensure Snowflake integration for live data analysis
5. **Test Agent Performance**: Validate execution time and report quality
6. **Update Documentation**: Include agent capabilities and workflow description

### Adding New Clinical Pattern Analysis

To extend the Clinical Events Explorer:

1. **Create Analysis Function**: Follow the pattern in `analyze_clinical_patterns_with_llm`
2. **Implement Progress Tracking**: Use the step-by-step progress update pattern
3. **Add Cortex Integration**: Ensure proper LLM model integration
4. **Test with Real Data**: Validate with actual clinical events data
5. **Document New Capabilities**: Update README with new analysis features

## Support

For technical support:

- **Application Issues**: Review troubleshooting section and create repository issue
- **AI Agent Problems**: Check agent-specific troubleshooting guide above
- **Clinical Pattern Analysis**: Review Clinical Events Explorer troubleshooting section
- **Snowflake Issues**: Check [Snowflake Documentation](https://docs.snowflake.com) and [Cortex AI](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions) documentation
- **Streamlit in Snowflake**: Visit [Streamlit in Snowflake documentation](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- **Fivetran**: Check [Fivetran documentation](https://fivetran.com/docs) for data connector issues

## Acknowledgments

Built with:
- **Streamlit in Snowflake** for native cloud application development and agent orchestration
- **Snowflake Data Cloud** for healthcare data warehousing and real-time agent data access
- **Snowflake Cortex AI** for advanced AI/ML capabilities and autonomous clinical reasoning
- **Fivetran** for automated data movement and real-time synchronization
- **Altair** for statistical visualizations and agent performance dashboards

## Version History

### Streamlit in Snowflake Release v2.0 (Current)
- **Enhanced Clinical Events Explorer** with AI pattern analysis and step-by-step progress tracking
- **Improved Agent Status Management** with visual indicators and reliable start/stop controls
- **Advanced Settings Dashboard** with agent configuration, data quality monitoring, and performance tracking
- **Enhanced Agent Configuration** with customizable thresholds and analysis parameters
- **Data Quality Monitoring** with real-time assessment across all healthcare tables
- **Agent Performance Monitoring** with comprehensive execution metrics and analytics
- **Report Export & Management** with multiple format support and professional documentation
- **Application Maintenance Tools** with schema inspector and troubleshooting support
- **Model Selection per Agent** with individual Cortex model configuration
- **Previous Results Display** showing completed agent steps and historical reports

### Streamlit in Snowflake Release v1.0
- **Added Five Autonomous Healthcare Agents** with multi-step reasoning workflows
- **Implemented "Passing of the Baton"** workflow orchestration
- **Enhanced Settings Dashboard** with agent configuration and monitoring
- **Added Agent Performance Tracking** and optimization metrics
- **Implemented Data Quality Monitoring** for real-time validation
- **Added Report Export Capabilities** for clinical documentation
- **Native Snowflake Integration** with Cortex AI and single-file architecture
- **Optimized Performance** for Snowflake's compute and storage separation

### Complete Feature Summary

**All Five AI Agents Included:**
1. **🏥 Proactive Care Management Agent**
2. **👥 Population Health Manager Agent**
3. **💰 Prior Authorization Agent**
4. **🛡️ Quality & Safety Surveillance Agent**
5. **⚙️ Operations Optimization Agent**

**Complete Feature Set:**
- **AI Clinical Agents Tab**: All 5 autonomous agents with multi-step workflows
- **Patient Explorer**: Risk distribution, demographics, geographic analysis
- **Encounter Analysis**: Readmission tracking, encounter distribution
- **Clinical Events Explorer**: Event type analysis, timeline visualization, AI pattern analysis
- **Claims Analysis**: Financial metrics, payment efficiency
- **Cortex Clinical Insights**: 5 clinical analysis micro-applications
- **Analysis History**: Complete audit trail
- **Settings**: Agent configuration, data quality monitoring, performance tracking

**Snowflake-Specific Optimizations:**
1. **Cortex Integration**: Uses `SNOWFLAKE.CORTEX.COMPLETE()` for all AI functionality
2. **Data Type Handling**: Proper conversion of VARCHAR dates and numeric amounts
3. **Session Management**: Uses `get_active_session()` for Snowflake connectivity
4. **Schema Queries**: Adapted to Snowflake INFORMATION_SCHEMA
5. **Boolean Handling**: Handles Snowflake VARCHAR boolean fields properly

**Database Configuration:**
- **Database**: `HOL_DATABASE`
- **Schema**: `PATIENT_RECIDIVISM_HEALTHCARE`
- **Tables**: `patients`, `encounters`, `clinical_events`, `claims`, `risk_scores`

---

**🚀 Experience the future of healthcare analytics with autonomous AI agents that transform complex clinical workflows into actionable insights through intelligent automation and multi-step reasoning - all powered by Snowflake's native AI capabilities with enhanced clinical pattern analysis and comprehensive monitoring.** 