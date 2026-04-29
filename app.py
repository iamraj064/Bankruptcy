import sqlite3
import logging
import sys
import json
import re
import pandas as pd
import streamlit as st
from config import call_llm, call_llm_haiku
from dotenv import load_dotenv
from insights_generator import generate_insights

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bankruptcy_genbi.log", encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("bankruptcy_genbi")

st.set_page_config(
    page_title="Bankruptcy GenBI Assistant",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    .stSuccess {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        color: #065f46;
    }
    .stError {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1>📊 Bankruptcy GenBI Assistant</h1>",
    unsafe_allow_html=True,
)


@st.cache_data
def load_schema():
    """Load database schema from schema.json"""
    try:
        with open('schema.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load schema.json: %s", e)
        return None


def get_actual_database_schema():
    """Get the actual column structure from the uploaded_data table in database"""
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(uploaded_data);")
        columns_info = cursor.fetchall()
        conn.close()
        
        if not columns_info:
            logger.warning("uploaded_data table not found in database")
            return None
        
        # Build schema from actual database columns
        actual_columns = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2] or "text"
            
            # Try to match with schema.json for description, fallback to generic
            description = f"Column {col_name}"
            
            actual_columns.append({
                "name": col_name,
                "type": col_type,
                "description": description
            })
        
        logger.info("Detected %d columns from uploaded_data table", len(actual_columns))
        return {
            "schema_version": "1.0",
            "table_name": "uploaded_data",
            "description": "Dynamic schema from uploaded CSV data",
            "columns": actual_columns
        }
    except Exception as e:
        logger.exception("Error getting actual database schema: %s", e)
        return None
    
def extract_sql_from_response(text: str) -> str:
    """Extract SQL query from LLM response in various formats"""
    try:
        text = text.strip()
        # First, try to find a JSON object
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and 'sql' in obj:
                    logger.debug("SQL extracted from JSON object (sql key)")
                    return obj['sql'].strip()
                else:
                    if isinstance(obj, dict) and 'CORRECTED_QUERY' in obj:
                        logger.debug("SQL extracted from JSON object (CORRECTED_QUERY key)")
                        return obj['CORRECTED_QUERY'].strip()
            except Exception as e:
                logger.debug("Failed to parse JSON: %s", e)
                pass

        # If no JSON, try to find SQL fenced block
        m = re.search(r"```sql\n(.*?)```", text, re.S | re.I)
        if m:
            logger.debug("SQL extracted from code fence block")
            return m.group(1).strip()

        # Fallback: attempt to extract first SELECT ... statement
        m = re.search(r"(SELECT[\s\S]*?);?\s*$", text, re.I)
        if m:
            logger.debug("SQL extracted from SELECT statement")
            return m.group(1).strip()

        logger.warning("No SQL could be extracted from response")
        return ""
    except Exception as e:
        logger.exception("Error extracting SQL from response: %s", e)
        return ""

# def follow_up_question(user_question, conversation_memory=None):
#     """
#     Robustly answers follow-up questions by using the LLM to extract logic 
#     and Python to perform the actual math/filtering on the dataset.
#     """
#     try:
#         logger.info("Processing follow-up question with Python engine: %s", user_question)
        
#         # 1. Retrieve the latest result
#         last_result = st.session_state.get("last_result")
#         if not last_result or not last_result.get("records"):
#             return "I don't have the data from the previous query to answer this. Please run a new search first."

#         records = last_result["records"]
#         df = pd.DataFrame(records)
#         cols_info = "\n".join([f"- {col} (Type: {df[col].dtype})" for col in df.columns])

#         # 2. Use LLM to extract the filter logic
#         prompt = (
#             "You are a Logic Extraction Agent. Translate the user question into a JSON filtering rule.\n\n"
#             "COLUMNS IN DATASET:\n"
#             f"{cols_info}\n\n"
#             "USER QUESTION:\n"
#             f"'{user_question}'\n\n"
#             "INSTRUCTIONS:\n"
#             "1. Identify the column, operator (> , < , == , >= , <= , max, min), and target value.\n"
#             "2. Return ONLY a JSON object. No conversational text.\n\n"
#             "FORMAT EXAMPLE:\n"
#             "{\"column\": \"count\", \"op\": \">\", \"value\": 5, \"intent\": \"filter\"}\n\n"
#             "JSON OUTPUT:"
#         )
        
#         response = call_llm_haiku(prompt)
        
#         # 3. Parse logic and execute in Python
#         import json
#         import re
#         try:
#             # Robust JSON extraction using Regex
#             json_match = re.search(r"(\{.*\})", response.strip(), re.DOTALL)
#             if not json_match:
#                 logger.error("No JSON found in LLM response: %s", response)
#                 return "I couldn't extract the calculation logic. Please try again with a simple question like 'count > 5'."
            
#             logic = json.loads(json_match.group(1))
#             col = logic.get("column")
#             op = logic.get("op")
#             val = logic.get("value")
#             intent = logic.get("intent", "filter")
            
#             # Case-insensitive column matching
#             actual_col = next((c for c in df.columns if c.lower() == str(col).lower()), None)
            
#             if not actual_col:
#                 return f"I couldn't find the column '{col}' in the current result set. Available: {', '.join(df.columns)}"

#             # Robust type conversion for value
#             try:
#                 if val is not None:
#                     # Try to match the type of the column
#                     if pd.api.types.is_numeric_dtype(df[actual_col]):
#                         val = float(val)
#             except:
#                 pass

#             # Perform the math in Python
#             if intent == "filter":
#                 if op == ">": filtered_df = df[df[actual_col] > val]
#                 elif op == "<": filtered_df = df[df[actual_col] < val]
#                 elif op == ">=": filtered_df = df[df[actual_col] >= val]
#                 elif op == "<=": filtered_df = df[df[actual_col] <= val]
#                 elif op in ["==", "="]: filtered_df = df[df[actual_col] == val]
#                 else: filtered_df = df # Fallback
#             elif intent == "rank" or op in ["max", "min"]:
#                 if op == "max" or "high" in str(user_question).lower(): 
#                     filtered_df = df[df[actual_col] == df[actual_col].max()]
#                 else: 
#                     filtered_df = df[df[actual_col] == df[actual_col].min()]
#             else:
#                 filtered_df = df

#             match_count = len(filtered_df)
            
#             # 4. Final summary
#             if match_count == 0:
#                 return f"I analyzed the records and found 0 results where **{actual_col}** is **{op} {val}**."
            
#             summary_items = filtered_df.to_dict(orient='records')
#             result_text = f"**Found {match_count} matches:**\n\n"
#             for item in summary_items[:20]: # Show up to 20
#                 row_str = " | ".join([f"**{k}**: {v}" for k, v in item.items()])
#                 result_text += f"- {row_str}\n"
            
#             if match_count > 20:
#                 result_text += f"\n*(Showing first 20 of {match_count} records)*"
                
#             return result_text

#         except Exception as json_err:
#             logger.error("Execution error: %s | Response: %s", json_err, response)
#             return "I understood your question but couldn't perform the calculation. Try something like: 'where count is greater than 5'."

#     except Exception as e:
#         logger.exception("Failure in follow_up_question: %s", e)
#         return "I encountered an error analyzing the data."

#     except Exception as e:
#         logger.exception("Failure in follow_up_question: %s", e)
#         return "I encountered an error analyzing the data."

# def follow_up_question(user_question, conversation_memory=None):
#     """
#     Advanced follow-up handler:
#     Supports filter, range, aggregate, and combined queries.
#     """
#     try:
#         logger.info("Processing follow-up question: %s", user_question)

#         # 1. Get previous result
#         last_result = st.session_state.get("last_result")
#         if not last_result or not last_result.get("records"):
#             return "I don't have previous data. Please run a query first."

#         df = pd.DataFrame(last_result["records"])

#         # 2. Column info
#         cols_info = "\n".join(
#             [f"- {col} (Type: {df[col].dtype})" for col in df.columns]
#         )

#         # 3. LLM Prompt (Expanded Schema)
#         prompt = f"""


# You are a Logic Extraction Agent.

# COLUMNS:
# {cols_info}

# USER QUESTION:
# "{user_question}"

# INSTRUCTIONS:
# - Identify intent: filter, aggregate, range, or combined
# - Operators:
#     filter: >, <, >=, <=, == 
#     aggregate: avg, sum, count, min, max
#     range: between
# - For combined queries, include both filter and aggregate
# - Return ONLY JSON

# EXAMPLES:
# {{"intent":"aggregate","column":"count","op":"avg"}}
# {{"intent":"filter","column":"count","op":">","value":5}}
# {{"intent":"range","column":"count","op":"between","value":[6,11]}}
# {{
#   "intent":"combined",
#   "filter":{{"column":"count","op":"between","value":[6,11]}},
#   "aggregate":{{"column":"year","op":"count"}}
# }}

# JSON OUTPUT:
# """

#         response = call_llm_haiku(prompt)

#         # 4. Extract JSON safely
#         import json, re

#         match = re.search(r"\{.*?\}", response, re.DOTALL)
#         if not match:
#             logger.error("No JSON found: %s", response)
#             return "Couldn't understand the query. Try simpler phrasing."

#         logic = json.loads(match.group(0))

#         # Helper: match column safely
#         def get_column(col_name):
#             return next(
#                 (c for c in df.columns if c.lower() == str(col_name).lower()),
#                 None,
#             )

#         # Helper: apply filter
#         def apply_filter(df, rule):
#             col = get_column(rule.get("column"))
#             op = rule.get("op")
#             val = rule.get("value")

#             if not col:
#                 raise ValueError(f"Column '{rule.get('column')}' not found")

#             if pd.api.types.is_numeric_dtype(df[col]):
#                 try:
#                     if isinstance(val, list):
#                         val = [float(v) for v in val]
#                     else:
#                         val = float(val)
#                 except:
#                     pass

#             if op == ">":
#                 return df[df[col] > val]
#             elif op == "<":
#                 return df[df[col] < val]
#             elif op == ">=":
#                 return df[df[col] >= val]
#             elif op == "<=":
#                 return df[df[col] <= val]
#             elif op in ["==", "="]:
#                 return df[df[col] == val]
#             elif op == "between" and isinstance(val, list) and len(val) == 2:
#                 return df[(df[col] >= val[0]) & (df[col] <= val[1])]
#             else:
#                 return df

#         # Helper: aggregate
#         def apply_aggregate(df, rule):
#             col = get_column(rule.get("column"))
#             op = rule.get("op")

#             if not col:
#                 raise ValueError(f"Column '{rule.get('column')}' not found")

#             if op == "avg":
#                 return f"The average of **{col}** is **{round(df[col].mean(), 2)}**."
#             elif op == "sum":
#                 return f"The sum of **{col}** is **{df[col].sum()}**."
#             elif op == "count":
#                 return f"There are **{df[col].count()}** records."
#             elif op == "max":
#                 return f"The maximum of **{col}** is **{df[col].max()}**."
#             elif op == "min":
#                 return f"The minimum of **{col}** is **{df[col].min()}**."
#             else:
#                 return "Unsupported aggregation."

#         intent = logic.get("intent", "filter")

#         # 5. Execute logic
#         if intent == "filter":
#             filtered_df = apply_filter(df, logic)
#             count = len(filtered_df)

#             if count == 0:
#                 return "No matching records found."

#             preview = filtered_df.head(20).to_dict(orient="records")
#             text = f"**Found {count} matches:**\n\n"

#             for row in preview:
#                 text += "- " + " | ".join([f"**{k}**: {v}" for k, v in row.items()]) + "\n"

#             if count > 20:
#                 text += f"\n*(Showing first 20 of {count})*"

#             return text

#         elif intent == "range":
#             filtered_df = apply_filter(df, logic)
#             return f"There are **{len(filtered_df)}** records in the specified range."

#         elif intent == "aggregate":
#             return apply_aggregate(df, logic)

#         elif intent == "combined":
#             filter_rule = logic.get("filter")
#             agg_rule = logic.get("aggregate")

#             filtered_df = apply_filter(df, filter_rule)

#             if len(filtered_df) == 0:
#                 return "No data found after applying filter."

#             return apply_aggregate(filtered_df, agg_rule)

#         else:
#             return "Unsupported query type."

#     except Exception as e:
#         logger.exception("Error in follow_up_question: %s", e)
#         return "Something went wrong while analyzing the data."

def follow_up_question(user_question, conversation_memory=None):
    """
    Deterministic analytics engine with structured output.
    Returns clean JSON-safe responses for UI rendering.
    """

    import json
    import re
    import pandas as pd

    try:
        logger.info("Processing question: %s", user_question)

        # =====================================
        # 1. LOAD DATA
        # =====================================
        last_result = st.session_state.get("last_result")

        if not last_result or not last_result.get("records"):
            return {"type": "text", "message": "No previous data available."}

        df = pd.DataFrame(last_result["records"])

        if df.empty:
            return {"type": "text", "message": "Dataset is empty."}

        # =====================================
        # 2. PRE-PARSE (DETERMINISTIC BOOST)
        # =====================================
        q = user_question.lower()

        forced = {
            "count": any(x in q for x in ["how many", "count", "number of"]),
            "top_n": "top" in q,
            "avg": any(x in q for x in ["average", "mean"]),
            "sum": any(x in q for x in ["sum", "total"]),
        }

        # =====================================
        # 3. LLM PARSING
        # =====================================
        cols_info = "\n".join([f"- {c} ({df[c].dtype})" for c in df.columns])

        prompt = f"""
Extract structured query in JSON.

COLUMNS:
{cols_info}

QUESTION:
"{user_question}"

Return ONLY valid JSON.

SUPPORTED intents:
filter | range | aggregate | top_n | group_by | combined

EXAMPLES:
{{"intent":"filter","column":"count","op":"<","value":10}}
{{"intent":"aggregate","column":"count","op":"avg"}}
{{"intent":"top_n","column":"count","n":5,"order":"desc"}}
{{"intent":"group_by","group_by":"year","metric":{{"column":"count","op":"avg"}}}}

JSON:
"""

        response = call_llm_haiku(prompt)

        # safer JSON extraction
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return {"type": "text", "message": "Couldn't understand query."}

        logic = json.loads(match.group(0))

        # =====================================
        # HELPERS
        # =====================================
        def get_col(name):
            return next((c for c in df.columns if c.lower() == str(name).lower()), None)

        def apply_filter(data, rule):
            if not rule:
                return data

            col = get_col(rule.get("column"))
            op = rule.get("op")
            val = rule.get("value")

            if not col:
                raise ValueError(f"Column '{rule.get('column')}' not found")

            try:
                if isinstance(val, list):
                    val = [float(v) for v in val]
                else:
                    val = float(val)
            except:
                pass

            if op == ">":
                return data[data[col] > val]
            elif op == "<":
                return data[data[col] < val]
            elif op == ">=":
                return data[data[col] >= val]
            elif op == "<=":
                return data[data[col] <= val]
            elif op in ["=", "=="]:
                return data[data[col] == val]
            elif op == "between":
                return data[(data[col] >= val[0]) & (data[col] <= val[1])]

            return data

        def aggregate(data, col, op):
            if op == "avg":
                return data[col].mean()
            elif op == "sum":
                return data[col].sum()
            elif op == "min":
                return data[col].min()
            elif op == "max":
                return data[col].max()
            elif op == "count":
                return data[col].count()

        def format_table(data, message):
            return {
                "type": "table",
                "data": data.to_dict(orient="records"),
                "columns": list(data.columns),
                "message": message
            }

        def format_text(message):
            return {
                "type": "text",
                "message": message
            }

        # =====================================
        # 4. EXECUTION
        # =====================================
        intent = logic.get("intent")

        # ---------- FILTER ----------
        if intent == "filter":
            filtered = apply_filter(df, logic)

            if filtered.empty:
                return format_text("No matching records found.")

            msg = f"There are {len(filtered)} records matching your criteria."
            return format_table(filtered.head(50), msg)

        # ---------- RANGE ----------
        elif intent == "range":
            filtered = apply_filter(df, {
                "column": logic.get("column"),
                "op": "between",
                "value": logic.get("value")
            })

            if filtered.empty:
                return format_text("No records found in range.")

            msg = f"There are {len(filtered)} records in the specified range."
            return format_table(filtered, msg)

        # ---------- AGGREGATE ----------
        elif intent == "aggregate":
            col = get_col(logic.get("column"))

            if not col:
                return format_text("Invalid column.")

            op = logic.get("op") or ("avg" if forced["avg"] else "sum" if forced["sum"] else None)

            val = aggregate(df, col, op)

            return format_text(f"{op.upper()} of {col} = {round(val, 2)}")

        # ---------- TOP N ----------
        elif intent == "top_n" or forced["top_n"]:
            col = get_col(logic.get("column"))

            if not col:
                return format_text("Invalid column.")

            n = logic.get("n", 5)
            order = logic.get("order", "desc")

            result = df.sort_values(col, ascending=(order == "asc")).head(n).reset_index(drop=True)
            result.index += 1

            return format_table(result, f"Top {n} records by {col}")

        # ---------- GROUP BY ----------
        elif intent == "group_by":
            group_col = get_col(logic.get("group_by"))
            metric = logic.get("metric")

            if not group_col or not metric:
                return format_text("Invalid group by query.")

            m_col = get_col(metric.get("column"))
            op = metric.get("op")

            result = df.groupby(group_col)[m_col].agg(op).reset_index()

            return format_table(result, f"{op.upper()} of {m_col} grouped by {group_col}")

        # ---------- COMBINED ----------
        elif intent == "combined":
            filtered = apply_filter(df, logic.get("filter"))

            if filtered.empty:
                return format_text("No data after filtering.")

            if forced["count"]:
                return format_text(f"There are {len(filtered)} records after filtering.")

            if logic.get("group_by"):
                group_col = get_col(logic.get("group_by"))
                metric = logic.get("metric")

                result = filtered.groupby(group_col)[
                    get_col(metric.get("column"))
                ].agg(metric.get("op")).reset_index()

                return format_table(result, "Filtered + grouped result")

            else:
                metric = logic.get("aggregate")

                val = aggregate(
                    filtered,
                    get_col(metric.get("column")),
                    metric.get("op")
                )

                return format_text(f"Result after filter = {round(val, 2)}")

        # ---------- FALLBACK ----------
        return format_text("Unsupported query type.")

    except Exception as e:
        logger.exception("Error: %s", e)
        return {"type": "text", "message": "Error processing query."}


    

def generate_sql_from_question(user_question, schema, conversation_memory=None):
    """Use LLM to generate SQL from natural language question"""
    try:
        logger.info("Generating SQL from user question: %s", user_question)
        prompt = (
            "You are an assistant that converts a natural language request into a single SQLite-compatible SQL query.\n"
            "All the dates are in the format YYYY-MM-DD.\n"
            "Return only valid JSON with a single key `sql` whose value is the SQL string.\n"
            "for date relates question Use Open_date ,Close_date column and formulate sql like strftime('%Y', Open_date) = '2024'\n"
            "Do NOT include explanations or additional fields. Ensure the SQL is compatible with SQLite.\n\n"
            "Database schema (JSON):\n" + json.dumps(schema['columns']) + "\n\n"
            "Table name: " + schema['table_name'] + "\n"
            "User request: \"" + user_question + "\"\n\n"
            "If the request cannot be represented as a single SQL SELECT query and do not include ';' at the end of the query, return an empty string for `sql`."
        )
        response = call_llm(prompt)
        sql_query = extract_sql_from_response(response)
        logger.info("SQL generated | length=%d | query=%s", len(sql_query), sql_query[:100] if sql_query else "EMPTY")
        return sql_query
    except Exception as e:
        logger.exception("Error generating SQL from question: %s", e)
        return ""




def validate_sql_with_judge(sql_query, user_question, schema):
    """Use an LLM as a judge to validate if the generated SQL is correct"""
    
    # Step 1: Check for DDL and DML queries - BLOCK these operations
    sql_upper = sql_query.strip().upper()
    
    # DDL (Data Definition Language) keywords - CREATE, ALTER, DROP, TRUNCATE, RENAME
    ddl_keywords = ['CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'RENAME']
    # DML (Data Manipulation Language) keywords - INSERT, UPDATE, DELETE, MERGE
    dml_keywords = ['INSERT', 'UPDATE', 'DELETE', 'MERGE']
    
    # Check if the query starts with or contains DDL/DML keywords
    for keyword in ddl_keywords + dml_keywords:
        if sql_upper.startswith(keyword) or re.search(r'\b' + keyword + r'\b', sql_upper):
            operation_type = "DDL" if keyword in ddl_keywords else "DML"
            logger.warning("Blocked %s operation: %s", operation_type, sql_query)
            return {
                "is_valid": False,
                "status": "BLOCKED",
                "explanation": f"{operation_type} queries are not allowed. Only SELECT queries are permitted.",
                "repaired_query": None,
            }
    
    # Step 2: Validate that query is a SELECT query
    if not sql_upper.startswith('SELECT'):
        logger.warning("Non-SELECT query attempted: %s", sql_query)
        return {
            "is_valid": False,
            "status": "INVALID",
            "explanation": "Only SELECT queries are allowed. The query must start with SELECT.",
            "repaired_query": None,
        }
    
    # Step 3: Use LLM to validate column names, syntax, and correctness
    try:
        columns_desc = "\n".join([
            f"- {col['name']} ({col['type']}): {col['description']}"
            for col in schema['columns']
        ])
        
        prompt = f"""You are an expert SQLite3 database validator. Your task is to judge whether the following SQL query is correct and valid. And treat each row in the table as one backstop.

DATABASE SCHEMA:
Table: {schema['table_name']}
Columns:
{columns_desc}

USER QUESTION: {user_question}

SQL QUERY TO VALIDATE:
{sql_query}

VALIDATION RULES:
1. Check if the SQL syntax is valid SQLite3
2. Check if column names are exact matches from the schema (no typos or incorrect column names)
3. Check if the query answers the user's question
4. Check if the date formats are correct (YYYY-MM-DD) if dates are involved.
5. Table name must be '{schema['table_name']}'
6. Do not include ';' at the end of the query
7. Reject any queries that attempt to modify data (INSERT, UPDATE, DELETE are already blocked)

RESPOND WITH ONLY valid JSON:
- If VALID: {{"VALID": "YES"}}
- If INVALID: {{"VALID": "NO", "CORRECTED_QUERY": "SELECT * FROM {schema['table_name']} WHERE severity = 'High'"}}

Do not include any explanations, text, or additional content outside the JSON object."""

        response = call_llm(prompt)
        logger.info("SQL Validation Response:\n%s", response)
        
        # Parse the response robustly
        is_valid = False
        explanation = "Validation response received"
        query = None
        
        response_text = response.strip()
        
        if 'YES' in response_text.upper():
            explanation = "Query is valid"
            is_valid = True
        else:
            explanation = "❌ Query is invalid"
            query = extract_sql_from_response(response_text)
        
        return {
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID",
            "explanation": explanation,
            "repaired_query": query if not is_valid else None,
        }
    except Exception as e:
        logger.exception("Error validating SQL with judge: %s", e)
        return {
            "is_valid": False,
            "status": "ERROR",
            "explanation": f"❌ Validation error: {str(e)}",
            "suggestion": None,
            "repaired_query": None
        }

def clean_sql_for_whitespace(sql_query):
    """Add TRIM() to WHERE clauses to handle whitespace in data"""
    try:
        import re
        
        # Pattern to match WHERE conditions with string values
        # Matches: column_name = 'value' or column_name = "value"
        pattern = r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\'"])([^\2]*?)\2'
        
        def replace_with_trim(match):
            col_name = match.group(1)
            quote = match.group(2)
            value = match.group(3)
            # Return the same but with TRIM() wrapped around column
            return f"WHERE TRIM({col_name}) = {quote}{value}{quote}"
        
        modified_query = re.sub(pattern, replace_with_trim, sql_query, flags=re.IGNORECASE)
        
        if modified_query != sql_query:
            logger.info("Added TRIM() to WHERE clause for whitespace handling | original_length=%d | modified_length=%d", len(sql_query), len(modified_query))
        
        return modified_query
    except Exception as e:
        logger.exception("Error cleaning SQL for whitespace: %s", e)
        return sql_query


def execute_sql_query(sql_query, table_name):
    """Execute SQL query against the database and return results"""
    try:
        # Log the original SQL for debugging
        logger.info("Original SQL query: %s", sql_query)
        
        # Clean up SQL for whitespace issues
        sql_query = clean_sql_for_whitespace(sql_query)
        logger.info("Executing SQL query (after cleanup): %s", sql_query)
        
        # Validate that query references the correct table
        if table_name.lower() not in sql_query.lower():
            logger.warning("Generated SQL doesn't reference the correct table. Adding FROM clause.")

        # Get actual columns from database to validate the query
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(uploaded_data);")
        db_columns = [row[1] for row in cursor.fetchall()]
        
        # Note: We skip column validation for aggregation queries (COUNT, SUM, etc)
        # as they don't reference actual columns in the same way
        is_aggregation_query = any(keyword in sql_query.upper() for keyword in ['COUNT(*)', 'COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
        
        if not is_aggregation_query:
            # Extract column names from the SQL query for validation
            import re
            
            # Look for column references in SELECT clause and WHERE/AND/OR conditions
            select_columns = re.findall(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
            where_columns = re.findall(r'WHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]', sql_query)
            
            potential_columns = []
            
            # Extract from SELECT clause (if found)
            if select_columns:
                # Split by comma and clean up each column
                cols = select_columns[0].split(',')
                for col in cols:
                    col = col.strip()
                    # Extract column name (handle aliases like "col AS alias")
                    col_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', col)
                    if col_match:
                        potential_columns.append(col_match.group(1))
            
            # Add WHERE clause columns
            potential_columns.extend(where_columns)
            potential_columns = list(set(potential_columns))
            
            # Table names, common aliases, and placeholder words to exclude
            exclude_names = {'uploaded_data', 'data', 'table', 'db', 'a', 'b', 'c', 't', 'u', 'd', 'Column', 'column', 'name', 'value', 'result', 'row', 'query', 'COUNT', 'Sum', 'Count', 'Average'}
            
            # Check for SQL keywords that might be in the regex results
            sql_keywords = {'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'LIMIT', 'ORDER', 'BY', 'ASC', 'DESC', 'GROUP', 'HAVING', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CAST', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON', 'JOIN', 'LEFT', 'INNER', 'OUTER', 'CROSS', 'LIKE', 'IN', 'IS', 'NOT', 'NULL', 'BETWEEN', 'INSERT', 'UPDATE', 'DELETE'}
            
            # Filter out SQL keywords and excluded names
            referenced_columns = [col for col in potential_columns if col not in sql_keywords and col not in exclude_names and col.upper() not in sql_keywords]
            
            # Check if all referenced columns exist in the database
            missing_columns = [col for col in referenced_columns if col not in db_columns]
            
            if missing_columns:
                error_msg = f"Column(s) {missing_columns} do not exist in the database. Available columns: {db_columns}"
                logger.error(error_msg)
                conn.close()
                st.error(f"**Column Validation Error:** {error_msg}")
                return None
        
        # Execute query
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        logger.info("SQL query executed successfully | rows=%d", len(df))
        
        # If query returned 0 rows, show debugging info
        if len(df) == 0:
            logger.warning("Query returned 0 rows. Showing sample data and debug info...")
            try:
                conn = sqlite3.connect('data.db')
                # Show sample data
                sample_df = pd.read_sql_query("SELECT * FROM uploaded_data LIMIT 5;", conn)
                logger.info("Sample data from database:\n%s", sample_df)
                
                # Extract WHERE condition to help debug
                import re
                where_match = re.search(r'WHERE\s+(.+?)(?:LIMIT|;|$)', sql_query, re.IGNORECASE)
                if where_match:
                    where_clause = where_match.group(1).strip()
                    # Try to extract column name and value
                    col_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*[\'"]?([^\'"]+)[\'"]?', where_clause)
                    if col_match:
                        col_name = col_match.group(1)
                        col_value = col_match.group(2)
                        logger.info("WHERE clause analysis: column='%s', value='%s'", col_name, col_value)
                        
                        # Show distinct values for this column
                        try:
                            distinct_df = pd.read_sql_query(f"SELECT DISTINCT {col_name} FROM uploaded_data LIMIT 10;", conn)
                            logger.info("Sample distinct values for '%s':\n%s", col_name, distinct_df)
                        except Exception as e:
                            logger.warning("Could not fetch distinct values: %s", e)
                
                conn.close()
            except Exception as e:
                logger.warning("Error during debug analysis: %s", e)
        
        return df
    except Exception as e:
        logger.exception("Error executing SQL query: %s", e)
        st.error(f"**SQL Execution Error:** {str(e)}")
        return None
    
def should_generate_insights(user_query, result_df):
    """Generate insights only when user intent explicitly asks for it."""
    if result_df is None or result_df.empty or len(result_df) < 2:
        logger.debug("Skipping insights generation: empty or insufficient data | rows=%d", len(result_df) if result_df is not None else 0)
        return False

    insight_keywords = {
        "insight",
        "insights",
        "plot",
        "plots",
        "chart",
        "charts",
        "graph",
        "graphs",
        "visual",
        "visualize",
        "visualise",
        "visualization",
        "distribution",
        "trend",
        "breakdown",
        "compare",
        "comparison",
        "percentage",
        "share",
        "dashboard",
    }
    query_lower = user_query.lower()
    should_generate = any(keyword in query_lower for keyword in insight_keywords)
    logger.debug("Insights generation check | should_generate=%s | keywords_found=%s", should_generate, 
                 [k for k in insight_keywords if k in query_lower])
    return should_generate

def _initialize_conversation_memory():
    """Initialize conversation memory structure"""
    memory = {
        "history": [],
        "last_user_query": None,
        "last_assistant_response": None,
    }
    logger.info("Conversation memory initialized")
    return memory


def _append_conversation_memory(user_query: str, sql_query: str, records: list, validation_result: dict = None, answer: str = None):
    """
    Append structured conversation memory with user question, SQL query, and records.
    
    Args:
        user_query: The user's natural language question
        sql_query: The generated SQL query
        records: List of dictionaries containing fetched records
        validation_result: Optional validation result dictionary
        answer: The assistant's response to the user's query
    """
    try:
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = _initialize_conversation_memory()

        memory = st.session_state.conversation_memory
        
        # Build structured memory entry
        if not answer:
            memory_entry = {
                "user_question": user_query,
                "sql_query": sql_query,
                "records": records,
                "record_count": len(records) if records else 0,
            }
        else:
            memory_entry = {
                "user_question": user_query,
                "assistant_answer": answer,
            }
        
        # Add validation result if provided
        if validation_result:
            memory_entry["validation_result"] = validation_result.get("status", "UNKNOWN")
        
        memory["history"].append(memory_entry)
        memory["last_user_query"] = user_query
        if not answer:
            memory["last_assistant_response"] = f"SQL Query: {sql_query} | Total Number of Records: {len(records) if records else 0} | Records: {records}"
        else:
            memory["last_assistant_response"] = answer

        logger.info("Conversation memory appended | query_length=%d | records=%d | history_length=%d", 
                   len(user_query), len(records) if records else 0, len(memory["history"]))

        # Keep only last 10 conversations
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]
            logger.debug("Conversation memory trimmed to last 10 entries")
    except Exception as e:
        logger.exception("Error appending conversation memory: %s", e)


def _build_memory_context() -> str:
    """Build context string from conversation memory for LLM context."""
    try:
        memory = st.session_state.get("conversation_memory")
        if not memory or not memory.get("history"):
            logger.debug("No conversation history found")
            return "No prior conversation context."
        logger.debug("Memory context built from %d conversation entries", len(memory.get("history", [])))
        return memory
    except Exception as e:
        logger.exception("Error building memory context: %s", e)
        return "No prior conversation context."

def _load_and_clean_csv(uploaded_file):
    """Load and clean CSV data - handles whitespace in column names and data"""
    try:
        df = pd.read_csv(uploaded_file)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        # Remove columns with empty names (columns that were only whitespace)
        df = df.loc[:, df.columns != '']
        
        # Clean data: strip whitespace from all string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        logger.info("CSV loaded and cleaned | filename=%s | shape=%s", uploaded_file.name, df.shape)
        return df
    except Exception as e:
        logger.exception("Error loading and cleaning CSV: %s", e)
        return None
    
def session_changer():
    """Function to trigger re-rendering of the chat when follow-up questions toggle is changed."""
    follow_up_enabled = st.session_state.follow_up_toggle_state
    st.session_state.follow_up_toggle = follow_up_enabled
    logger.info("Follow-up questions toggle changed | enabled=%s", follow_up_enabled)

def main():
    logger.info("=== Streamlit session started ===")
    
    # Initialize conversation memory if not present
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = _initialize_conversation_memory()
    
    # Load schema
    logger.info("Loading database schema")
    schema = load_schema()
    if not schema:
        logger.error("Failed to load schema - schema.json missing")
        st.error("Failed to load database schema. Please ensure schema.json exists.")
        return

    # Sidebar
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Initialize session state for tracking uploaded file
    if "last_uploaded_file_name" not in st.session_state:
        st.session_state.last_uploaded_file_name = None
    if "data_in_db" not in st.session_state:
        st.session_state.data_in_db = False
    if "actual_schema" not in st.session_state:
        st.session_state.actual_schema = None
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "follow_up_toggle" not in st.session_state:
        st.session_state.follow_up_toggle = False

    if uploaded_file is not None:
        # Check if this is a new file
        file_changed = uploaded_file.name != st.session_state.last_uploaded_file_name
        
        # Load and clean CSV data
        df = _load_and_clean_csv(uploaded_file)
        if df is None:
            logger.error("Failed to load CSV file: %s", uploaded_file.name)
            st.error(f"Could not process the uploaded file.")
            st.session_state.data_in_db = False
            return
        
        if file_changed:
            logger.info("New file detected | filename=%s", uploaded_file.name)
            try:
                st.sidebar.success(f"File `{uploaded_file.name}` loaded successfully!")
                st.sidebar.subheader("Data Preview:")
                st.sidebar.dataframe(df.head(), hide_index=True)
                
                # Auto-save to SQLite
                conn = sqlite3.connect('data.db')
                df.to_sql('uploaded_data', conn, if_exists='replace', index=False)
                conn.close()
                
                # Update session state
                st.session_state.last_uploaded_file_name = uploaded_file.name
                st.session_state.data_in_db = True
                
                # Get actual database schema
                st.session_state.actual_schema = get_actual_database_schema()
                
                st.sidebar.success(f"Saved to database | Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                logger.info("Data saved to SQLite | filename=%s | rows=%d | columns=%d", uploaded_file.name, df.shape[0], df.shape[1])
            except Exception as e:
                logger.exception("Failed to process and save uploaded file: %s", uploaded_file.name)
                st.error(f"Could not process the uploaded file. (Detail: {e})")
                st.session_state.data_in_db = False
                return
        else:
            logger.debug("Same file already loaded | filename=%s", uploaded_file.name)
            # Same file, show cached data
            st.sidebar.success(f"File `{uploaded_file.name}` loaded")
            st.sidebar.subheader("Data Preview:")
            st.sidebar.dataframe(df.head(), hide_index=True)

    # Check if data exists in database
    if st.session_state.data_in_db:
        try:
            conn = sqlite3.connect('data.db')
            check_df = pd.read_sql_query("SELECT COUNT(*) as count FROM uploaded_data;", conn)
            conn.close()
            count = check_df['count'].values[0] if len(check_df) > 0 else 0
            if count > 0:
                logger.info("Database contains %d records", count)
                st.sidebar.info(f"{count} records available")
                
                # Show detected columns
                if st.session_state.actual_schema:
                    col_names = [col['name'] for col in st.session_state.actual_schema['columns']]
                    logger.debug("Detected %d columns in schema", len(col_names))
                    st.sidebar.caption(f"**Detected Columns:** {len(col_names)}")
                    with st.sidebar.expander("View all columns"):
                        st.write(col_names)
            else:
                logger.warning("Database is empty - no records found")
                st.session_state.data_in_db = False
        except Exception as e:
            logger.debug("Database check failed: %s", e)
            st.session_state.data_in_db = False

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello ! I can help you query your Bankruptcy database using natural language. Just ask me any question about your data!",
            }
        ]

    # Replay previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg.get("content")
            
            # Handle structured dictionary content (for follow-ups)
            if isinstance(content, dict):
                if content.get("type") == "table":
                    st.markdown(content.get("message", "Result:"))
                    df_res = pd.DataFrame(content.get("data", []))
                    st.dataframe(df_res, width='content', hide_index=True)
                elif content.get("type") == "text":
                    st.markdown(content.get("message", ""))
                else:
                    st.write(content)
            else:
                # Standard markdown content
                st.markdown(content)

            # Display dataframe if present (for standard queries)
            if msg["role"] == "assistant":
                if "sql_query" in msg:
                    with st.expander("Generated SQL"):
                        st.code(msg["sql_query"], language="sql")
                        
                if "validation_result" in msg:
                    validation_result = msg["validation_result"]
                    with st.expander("SQL Validation Report"):
                        st.markdown(f"**Status:** `{validation_result['status']}`")
                        st.markdown(f"**Explanation:** {validation_result['explanation']}")
                        if validation_result.get('suggestion'):
                            st.markdown(f"**Suggested Query:** ")
                            st.code(validation_result['suggestion'], language="sql")
                            
                        # If invalid, check if there was a repaired query
                        if not validation_result.get('is_valid') and validation_result.get('repaired_query'):
                            st.warning(f"SQL validation failed: {validation_result['explanation']}")
                            st.success("Auto-repaired SQL:")
                            st.code(validation_result['repaired_query'], language="sql")
                            
                # Display dataframe if present (for standard queries)
                if "dataframe" in msg:
                    st.dataframe(msg["dataframe"], width='content', hide_index=True)
                
                # Re-render insights/charts if they were originally generated
                if msg.get("has_insights") and msg.get("user_query"):
                    chart_type = msg.get("chart_type", "auto")
                    generate_insights(msg["dataframe"], chart_type=chart_type, user_query=msg["user_query"])

    # New user input
    st.sidebar.toggle("Follow-up questions", key="follow_up_toggle_state", on_change = session_changer)
    user_query = st.chat_input("Ask me anything about your data...")
    if not user_query:
        return

    # Check if we have data in database before processing query
    if not st.session_state.data_in_db:
        logger.warning("Query attempted without data | query=%s", user_query)
        st.warning("Please upload a CSV file first to query the database.")
        return

    logger.info("User query received | follow_up_mode=%s | query=%s", st.session_state.follow_up_toggle, user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Check if follow-up questions toggle is enabled
        if st.session_state.follow_up_toggle:
            logger.info("Processing as follow-up question")
            with st.spinner("Generating follow-up response..."):
                response = follow_up_question(user_query)
                
                # Render the response immediately based on type
                if isinstance(response, dict):
                    if response.get("type") == "table":
                        st.markdown(response.get("message", "Result:"))
                        df_res = pd.DataFrame(response.get("data", []))
                        st.dataframe(df_res, width='content', hide_index=True)
                    elif response.get("type") == "text":
                        st.markdown(response.get("message", ""))
                else:
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            _append_conversation_memory(user_query=user_query, answer = str(response), sql_query=None, records=None)
        else:
            logger.info("Processing as standard query")
            with st.spinner("Analyzing your question and generating query..."):
                _handle_user_query(user_query, schema)

def detect_chart_type(user_query):
    """Detect whether the user explicitly asked for a bar chart or pie chart."""
    try:
        query_lower = user_query.lower()

        pie_keywords = ["pie chart", "piechart", "pie", "donut"]
        bar_keywords = ["bar chart", "barchart", "bar plot", "barplot", "histogram", "column chart", "columnchart", "bar"]

        if any(keyword in query_lower for keyword in pie_keywords):
            logger.debug("Chart type detected: PIE")
            return "pie"
        if any(keyword in query_lower for keyword in bar_keywords):
            logger.debug("Chart type detected: BAR")
            return "bar"
        logger.debug("Chart type detected: AUTO")
        return "auto"
    except Exception as e:
        logger.exception("Error detecting chart type: %s", e)
        return "auto"

def _handle_user_query(user_query, schema):
    """Handle user queries by generating, validating, and executing SQL"""
    try:
        cache_key = user_query.strip().lower()
        
        # Check cache first
        if cache_key in st.session_state.query_cache:
            logger.info("Query found in cache | query=%s", user_query)
            st.success("Result fetched from cache")
            cached_data = st.session_state.query_cache[cache_key]
            
            sql_query = cached_data["sql_query"]
            result_df = cached_data["result_df"]
            
            # Convert dataframe to records for memory
            records = result_df.to_dict(orient='records') if len(result_df) > 0 else []
            _append_conversation_memory(user_query, sql_query, records)
            
            with st.expander("SQL Query"):
                st.code(sql_query, language="sql")
            if len(result_df) == 0:
                msg = "No records matched your criteria."
                st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                logger.info("Cached query returned 0 rows")
                return
                
            success_msg = f"Found {len(result_df)} records."
            st.success(success_msg)
            st.dataframe(result_df, width='content', hide_index=True)
            
            if should_generate_insights(user_query, result_df):
                chart_type = detect_chart_type(user_query)
                if chart_type != "auto":
                    st.info(f"Generating insights and {chart_type} chart based on your request...")
                else:
                    st.info("Generating  insights and visualizations from the result set...")
                generate_insights(result_df, chart_type=chart_type, user_query=user_query)
                
            _has_insights = should_generate_insights(user_query, result_df)
            assistant_entry = {
                "role": "assistant",
                "content": success_msg,
                "sql_query": sql_query,
                "dataframe": result_df,
                "user_query": user_query,
                "has_insights": _has_insights,
                "chart_type": detect_chart_type(user_query) if _has_insights else None
            }
            st.session_state.messages.append(assistant_entry)
            logger.info("Cached query executed successfully | rows=%d", len(result_df))
            return


        # Step 1: Generate SQL from natural language question
        logger.info("Starting SQL generation workflow | query=%s", user_query)
        st.info("Step 1: Generating SQL from your question...")
        sql_query = generate_sql_from_question(user_query, schema)
        
        if not sql_query:
            error_msg = "Failed to generate SQL query from your question. Please try rephrasing."
            logger.warning("SQL generation failed for query: %s", user_query)
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            _append_conversation_memory(user_query, "", [], None)
            return
        
        # Display generated SQL
        with st.expander("Generated SQL"):
            st.code(sql_query, language="sql")
        
        # Step 2: Validate SQL with judge LLM
        logger.info("Validating generated SQL")
        st.info("Step 2: Validating SQL ...")
        validation_result = validate_sql_with_judge(sql_query, user_query, schema)
        
        # Display validation result
        with st.expander("SQL Validation Report"):
            st.markdown(f"**Status:** `{validation_result['status']}`")
            st.markdown(f"**Explanation:** {validation_result['explanation']}")
        
        # Check if validation passed and use appropriate query
        final_query = sql_query
        
        if not validation_result['is_valid']:
            # Auto-repair was attempted
            if validation_result.get('repaired_query'):
                logger.warning("SQL validation failed, attempting auto-repair")
                st.warning(f"SQL validation failed: {validation_result['explanation']}")
                st.success("Auto-repaired SQL:")
                with st.expander("View Repaired Query"):
                    st.code(validation_result['repaired_query'], language="sql")
                final_query = validation_result['repaired_query']
            else:
                error_msg = f"SQL validation failed and auto-repair unsuccessful: {validation_result['explanation']}"
                logger.error("SQL validation failed with no repair possible: %s", error_msg)
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                _append_conversation_memory(user_query, sql_query, [], validation_result)
                return
        else:
            logger.info("SQL validation passed")
            st.success("SQL validation passed!")
        
        # Step 3: Execute the SQL query
        logger.info("Executing SQL query")
        st.info("Step 3: Executing query...")
        result_df = execute_sql_query(final_query, schema['table_name'])
        
        if result_df is None:
            error_msg = "Error executing the SQL query. Please check the database or try a different question."
            logger.error("SQL execution failed")
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            _append_conversation_memory(user_query, final_query, [], validation_result)
            return
        
        if len(result_df) == 0:
            msg = "Query executed successfully, but no records matched your criteria."
            logger.info("Query executed but returned 0 rows")
            st.info(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            _append_conversation_memory(user_query, final_query, [], validation_result)
            return
        
        # Update robust result memory for follow-up questions
        st.session_state.last_result = {
            "records": result_df.to_dict(orient='records'),
            "sql_query": final_query,
            "row_count": len(result_df)
        }
        
        # Display results
        success_msg = f"Query executed successfully! Found {len(result_df)} records."
        logger.info("Query executed successfully | rows=%d", len(result_df))
        st.success(success_msg)
        st.dataframe(result_df, width='content', hide_index=True)

        if should_generate_insights(user_query, result_df):
            chart_type = detect_chart_type(user_query)
            if chart_type != "auto":
                logger.info("Generating insights with %s chart", chart_type)
                st.info(f"Step 4: Generating insights and {chart_type} chart based on your request...")
            else:
                logger.info("Generating auto-detected insights")
                st.info("Step 4: Generating  insights and visualizations from the result set...")
            generate_insights(result_df, chart_type=chart_type, user_query=user_query)
        
        _has_insights = should_generate_insights(user_query, result_df)
        
        # Convert dataframe to records for memory storage
        records = result_df.to_dict(orient='records')
        
        assistant_entry = {
            "role": "assistant",
            "content": success_msg,
            "sql_query": final_query,
            "validation_result": validation_result,
            "dataframe": result_df,
            "user_query": user_query,
            "has_insights": _has_insights,
            "chart_type": detect_chart_type(user_query) if _has_insights else None
        }
        st.session_state.messages.append(assistant_entry)
        
        # Append conversation memory once with complete data
        _append_conversation_memory(user_query, final_query, records, validation_result)
        
        # Save to cache
        st.session_state.query_cache[user_query.strip().lower()] = {
            "sql_query": final_query,
            "result_df": result_df
        }
        logger.info("Query execution completed and cached | rows=%d | cache_size=%d", len(result_df), len(st.session_state.query_cache))
        
    except Exception as error:
        error_msg = f"Unexpected error: {str(error)}"
        logger.exception("Unexpected error in query handling: %s", error_msg)
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    try:
        logger.info("Starting Bankruptcy application")
        main()
    except Exception as e:
        logger.exception("Fatal error in main application: %s", e)
        raise
