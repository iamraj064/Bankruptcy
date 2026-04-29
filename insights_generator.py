import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
from config import call_llm

logger = logging.getLogger("insights_generator")

# ── Light professional chart theme (matches app UI) ─────────────────────────
PRIMARY   = "#2563EB"   # electric blue
ACCENT2   = "#7C3AED"   # violet
ACCENT3   = "#10B981"   # emerald
BG_DARK   = "#FFFFFF"
BG_PANEL  = "#a6d1f7"
TEXT_MAIN = "#1E293B"
TEXT_MUTED= "#64748B"
GRID_LINE = "#E2E8F0"

CHART_COLORS = [
    "#2563EB", "#7C3AED", "#10B981", "#F59E0B",
    "#EF4444", "#06B6D4", "#EC4899", "#84CC16",
    "#F97316", "#8B5CF6"
]

import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_DARK,
    "axes.edgecolor":    GRID_LINE,
    "axes.labelcolor":   TEXT_MAIN,
    "axes.titlecolor":   TEXT_MAIN,
    "axes.grid":         True,
    "axes.prop_cycle":   mpl.cycler(color=CHART_COLORS),
    "grid.color":        GRID_LINE,
    "grid.linewidth":    0.7,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  BG_DARK,
    "legend.edgecolor":  GRID_LINE,
    "legend.labelcolor": TEXT_MAIN,
    "figure.dpi":        110,
})
sns.set_style("whitegrid", {
    "axes.facecolor":  BG_DARK,
    "figure.facecolor": BG_DARK,
    "grid.color":      GRID_LINE,
})


class DataInsightsGenerator:
    """Generate comprehensive insights from query results"""
    
    def __init__(self, result_df: pd.DataFrame):
        self.df = result_df
        self.numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        self.categorical_cols = [col for col in self.df.columns if col not in self.numeric_cols]
        self.date_cols = self._detect_date_columns()
    
    def _detect_date_columns(self) -> List[str]:
        """Detect columns that represent dates"""
        date_cols = []
        for col in self.df.columns:
            if any(token in str(col).lower() for token in ["date", "time", "opened", "filed", "conversion", "status_date"]):
                try:
                    pd.to_datetime(self.df[col], errors="coerce")
                    if pd.to_datetime(self.df[col], errors="coerce").notna().sum() / len(self.df) >= 0.5:
                        date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset"""
        stats = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum()
        }
        return stats
    
    def get_numeric_insights(self) -> Dict:
        """Generate detailed insights for numeric columns"""
        insights = {}
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            insights[col] = {
                "count": len(col_data),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "skewness": col_data.skew(),
                "has_outliers": self._detect_outliers(col_data),
                "outlier_count": self._count_outliers(col_data)
            }
        
        return insights
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Detect if a series has outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).any()
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count number of outliers in a series"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def get_categorical_insights(self) -> Dict:
        """Generate detailed insights for categorical columns"""
        insights = {}
        
        for col in self.categorical_cols:
            col_data = self.df[col].astype(str).value_counts(dropna=True)
            
            insights[col] = {
                "unique_count": self.df[col].nunique(),
                "most_common": col_data.index[0] if len(col_data) > 0 else None,
                "most_common_freq": int(col_data.iloc[0]) if len(col_data) > 0 else 0,
                "least_common": col_data.index[-1] if len(col_data) > 0 else None,
                "least_common_freq": int(col_data.iloc[-1]) if len(col_data) > 0 else 0,
                "top_categories": col_data.head().to_dict(),
                "missing_count": self.df[col].isnull().sum()
            }
        
        return insights
    
    def detect_trends(self) -> Optional[Tuple[str, str]]:
        """Detect if data has date and numeric columns for trend analysis"""
        if self.date_cols and self.numeric_cols:
            return self.date_cols[0], self.numeric_cols[0]
        return None
    
    def get_correlations(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix for numeric columns"""
        if len(self.numeric_cols) >= 2:
            return self.df[self.numeric_cols].corr()
        return None
    
    def get_missing_data_report(self) -> Dict:
        """Generate report on missing data"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        report = {}
        for col in self.df.columns:
            if missing[col] > 0:
                report[col] = {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
        
        return report


class InsightVisualizer:
    """Handle visualization of insights with smart recommendations"""
    
    def __init__(self, result_df: pd.DataFrame, insights_gen: DataInsightsGenerator):
        self.df = result_df
        self.insights = insights_gen
    
    def _is_year_column(self, col_name: str, values: pd.Series) -> bool:
        """Detect whether a categorical column likely represents years"""
        col_lower = str(col_name).lower()
        if "year" in col_lower:
            return True

        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().all() and numeric_values.between(1900, 2100).all():
            return True

        parsed_dates = pd.to_datetime(values, errors="coerce", format="%Y")
        if parsed_dates.notna().all():
            return True

        return False

    def _sort_bar_data_by_category(self, bar_data: pd.DataFrame, cat_col: str, value_col: str) -> pd.DataFrame:
        """Sort chart categories chronologically when they represent years"""
        if self._is_year_column(cat_col, bar_data[cat_col]):
            bar_data = bar_data.copy()
            bar_data["_sort_key"] = pd.to_numeric(bar_data[cat_col], errors="coerce")
            if bar_data["_sort_key"].notna().all():
                return bar_data.sort_values("_sort_key").drop(columns=["_sort_key"])

        return bar_data.sort_values(value_col, ascending=True)

    def _build_llm_summary_prompt(
    self,
    stats: Dict,
    numeric_insights: Dict,
    categorical_insights: Dict,
        ) -> str:
        """Build a prompt for the LLM summarizing dataset trends and drivers."""
        prompt_lines = [
            "Role: You are an expert data analyst.",
            "Task: Provide a 3-5 liner sharp, practical summary of the dataset's key trends and drivers.",
            "",
            "Constraints:",
            "- NO raw statistics (mean, median, etc.).",
            "- NO technical jargon, SQL, code, or data quality reports.",
            "- NO generic filler or introductory fluff.",
            "- Format: Use plain English. **Bold** key findings and insights.",
            "- Length: Keep the response to a few concise sentences.",
            "",
            "Focus solely on what is driving the data and the most important business insights.",
            "",
        ]



        if self.insights.date_cols:
            prompt_lines.append(f"Detected date-like columns: {', '.join(self.insights.date_cols)}")

        if self.insights.numeric_cols:
            prompt_lines.append(f"Numeric columns: {', '.join(self.insights.numeric_cols)}")
            prompt_lines.append("Top numeric summaries:")
            for col, col_stats in list(numeric_insights.items())[:3]:
                prompt_lines.append(
                    f"- {col}: mean={round(col_stats['mean'], 2)}, median={round(col_stats['median'], 2)}, "
                    f"min={round(col_stats['min'], 2)}, max={round(col_stats['max'], 2)}, outliers={col_stats['outlier_count']}"
                )

        if self.insights.categorical_cols:
            prompt_lines.append(f"Categorical columns: {', '.join(self.insights.categorical_cols)}")
            prompt_lines.append("Top categorical distributions:")
            for col, cat_stats in list(categorical_insights.items())[:3]:
                top_categories = cat_stats.get('top_categories', {})
                category_summary = ', '.join([
                    f"{k} ({v})" for k, v in list(top_categories.items())[:3]
                ])
                prompt_lines.append(
                    f"- {col}: unique={cat_stats['unique_count']}, top={category_summary}"
                )

        if stats.get('missing_values'):
            missing_cols = [col for col, count in stats['missing_values'].items() if count > 0]
            if missing_cols:
                prompt_lines.append(
                    f"Columns with missing data: {', '.join(missing_cols)}"
                )

        prompt_lines.append("")
        prompt_lines.append(
            "Please summarize the most important patterns, trends, potential drivers, "
            "and data quality issues in this dataset."
        )
        prompt_lines.append(
            "If the dataset appears to have clear temporal trends, mention them. "
            "If the dataset is limited or sparse, note that too."
        )

        return "\n".join(prompt_lines)

    def _generate_llm_summary(self, stats: Dict) -> Optional[str]:
        """Generate an LLM-based executive summary of dataset insights."""
        try:
            numeric_insights = self.insights.get_numeric_insights()
            categorical_insights = self.insights.get_categorical_insights()
            prompt = self._build_llm_summary_prompt(stats, numeric_insights, categorical_insights)
            llm_response = call_llm(prompt, temperature=0.1)
            return llm_response.strip() if llm_response else None
        except Exception as e:
            logger.warning("LLM summary generation failed: %s", e)
            return None

    def render_executive_summary(self):
        """Render executive summary with key metrics"""
        stats = self.insights.generate_summary_statistics()

        # st.subheader("📊 Data Summary")

        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Total Records", stats["total_rows"])
        # with col2:
        #     st.metric("Columns", stats["total_columns"])
        # with col3:
        #     st.metric("Duplicates", stats["duplicate_rows"])
        # with col4:
        #     st.metric("Memory (MB)", stats["memory_usage_mb"])

        llm_summary = self._generate_llm_summary(stats)
        if llm_summary:
            st.subheader("🤖 Trend Summary Analysis")
            st.markdown(llm_summary)

        # Missing data report
        missing_report = self.insights.get_missing_data_report()
        if missing_report:
            with st.expander("⚠️ Missing Data Details", expanded=False):
                missing_df = pd.DataFrame([
                    {"Column": col, "Missing Count": data["count"], "% Missing": data["percentage"]}
                    for col, data in missing_report.items()
                ])
                st.dataframe(missing_df, use_container_width=True)
    
    def render_numeric_analysis(self):
        """Render comprehensive numeric column analysis"""
        numeric_insights = self.insights.get_numeric_insights()
        
        if not numeric_insights:
            return
        
        # st.subheader("🔢 Numeric Columns Analysis")
        
        # Summary statistics table
        # with st.expander("📈 Summary Statistics", expanded=True):
        #     stats_data = []
        #     for col, stats in numeric_insights.items():
        #         stats_data.append({
        #             "Column": col,
        #             "Count": stats["count"],
        #             "Mean": round(stats["mean"], 2),
        #             "Median": round(stats["median"], 2),
        #             "Std Dev": round(stats["std"], 2),
        #             "Min": round(stats["min"], 2),
        #             "Max": round(stats["max"], 2),
        #             "Outliers": stats["outlier_count"]
        #         })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Visualizations for key numeric columns
        key_numeric = self.insights.numeric_cols[:3]  # Focus on top 3 numeric columns
        
        if len(key_numeric) > 0:
            st.subheader("📊 Numeric Distributions (Bar Plot)")
            
            # Create bar plots for numeric distributions
            cols = st.columns(min(3, len(key_numeric)))
            
            for idx, col in enumerate(key_numeric):
                col_data = self.df[col].dropna()
                if len(col_data) < 2:
                    continue
                
                with cols[idx % 3]:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    
                    # Check if this is aggregated data (categorical + numeric pattern)
                    # If there's 1 categorical and 1 numeric column, show aggregated view
                    if len(self.insights.categorical_cols) >= 1 and len(self.insights.numeric_cols) == 1:
                        # This is aggregated data - show actual values with categories
                        cat_col = self.insights.categorical_cols[0]
                        
                        # Create data for bar chart
                        bar_data = self.df[[cat_col, col]].drop_duplicates()
                        bar_data = self._sort_bar_data_by_category(bar_data, cat_col, col)
                        
                        # Create bar plot with category labels
                        bars = ax.bar(range(len(bar_data)), bar_data[col].values,
                                     color=CHART_COLORS[0], alpha=0.95, edgecolor=BG_PANEL, linewidth=1.2)
                        
                        # Set X-axis to show category names
                        ax.set_xticks(range(len(bar_data)))
                        ax.set_xticklabels(bar_data[cat_col].values, rotation=45, ha='right', fontsize=9)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEXT_MAIN)
                        
                        # Set labels and title
                        ax.set_title(f"{col} by {cat_col}", fontsize=12, fontweight="bold")
                        ax.set_xlabel(cat_col, fontsize=10, fontweight='bold')
                        ax.set_ylabel(col, fontsize=10, fontweight='bold')
                    else:
                        # This is raw data - show distribution with binned frequencies
                        # Create bins and get counts
                        counts, bins = np.histogram(col_data, bins=15)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        
                        # Create bar plot
                        bars = ax.bar(range(len(counts)), counts, color=CHART_COLORS[0],
                                     alpha=0.99, edgecolor=BG_PANEL, linewidth=1.2)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEXT_MAIN)
                        
                        # Set labels and title
                        ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
                        ax.set_xlabel(f"{col} Value Ranges", fontsize=10, fontweight='bold')
                        ax.set_ylabel("Frequency (Count)", fontsize=10, fontweight='bold')
                    
                    ax.grid(axis="y", alpha=0.3, linestyle="--")
                    ax.set_axisbelow(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
    
    def render_correlation_analysis(self):
        """Render correlation analysis for numeric columns"""
        if len(self.insights.numeric_cols) < 2:
            return
        
        corr_matrix = self.insights.get_correlations()
        
        if corr_matrix is None or corr_matrix.empty:
            return
        
        st.subheader("🔗 Correlation Analysis")
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                    strong_corr.append({
                        "Variable 1": corr_matrix.columns[i],
                        "Variable 2": corr_matrix.columns[j],
                        "Correlation": round(corr_val, 3)
                    })
        
        # Display correlations
        if strong_corr:
            with st.expander("🔍 Strong Correlations (|r| > 0.5)", expanded=True):
                corr_df = pd.DataFrame(strong_corr).sort_values("Correlation", key=abs, ascending=True)
                st.dataframe(corr_df, use_container_width=True)
        
        # Heatmap
        with st.expander("📊 Correlation Heatmap", expanded=False):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                       cbar_kws={"label": "Correlation"}, ax=ax, square=True)
            ax.set_title("Numeric Variables Correlation Matrix", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
    
    def render_trend_analysis(self):
        """Render trend analysis if date and numeric columns exist"""
        trend_info = self.insights.detect_trends()
        
        if not trend_info:
            return
        
        date_col, numeric_col = trend_info
        
        st.subheader("📈 Trend Analysis")
        
        try:
            trend_df = self.df[[date_col, numeric_col]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
            trend_df = trend_df.dropna(subset=[date_col, numeric_col])
            trend_df = trend_df.sort_values(date_col)
            
            if len(trend_df) >= 2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(trend_df[date_col], trend_df[numeric_col], marker='o', linewidth=2.2,
                       color=PRIMARY, markersize=5, markerfacecolor=ACCENT2, markeredgecolor=PRIMARY)
                ax.set_xlabel(date_col)
                ax.set_ylabel(numeric_col)
                ax.set_title(f"{numeric_col} Over Time ({date_col})", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            logger.warning(f"Could not render trend analysis: {e}")
    
    def render_categorical_analysis(self):
        """Render categorical column analysis with pie charts only"""
        cat_insights = self.insights.get_categorical_insights()
        
        if not cat_insights:
            return
        
        # st.subheader("🏷️ Categorical Columns Analysis")
        
        # Focus on key categorical columns with reasonable cardinality
        key_categorical = [col for col in self.insights.categorical_cols 
                          if cat_insights[col]["unique_count"] <= 20][:3]
        
        if key_categorical:
            st.subheader("📊 Category Distributions (Pie Charts)")
            
            cols = st.columns(min(3, len(key_categorical)))
            
            for idx, col in enumerate(key_categorical):
                with cols[idx % 3]:
                    # Check if we have aggregated data (categorical + numeric pattern)
                    # If so, use the numeric column as values instead of value_counts()
                    
                    # For each categorical column, check if there's an associated numeric column
                    value_col = None
                    
                    # Strategy 1: If there's only 1 numeric column and 1-2 categorical columns, use the numeric one
                    if len(self.insights.numeric_cols) == 1 and len(self.insights.categorical_cols) <= 2:
                        value_col = self.insights.numeric_cols[0]
                    
                    # Strategy 2: Look for columns named "count", "total", "frequency", etc.
                    if value_col is None:
                        possible_value_cols = [c for c in self.insights.numeric_cols 
                                             if any(x in c.lower() for x in ['count', 'total', 'frequency', 'value', 'sum'])]
                        if possible_value_cols:
                            value_col = possible_value_cols[0]
                    
                    # Get the data for the pie chart
                    if value_col:
                        # Use aggregated data: categorical column values with numeric column as pie sizes
                        cat_data = self.df[[col, value_col]].drop_duplicates().sort_values(value_col, ascending=True)
                        pie_data = cat_data.set_index(col)[value_col]
                    else:
                        # Fall back to value_counts() for raw data
                        pie_data = self.df[col].astype(str).value_counts()
                    
                    if len(pie_data) == 0:
                        continue
                    
                    # PIE CHART with proper formatting
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = CHART_COLORS[:len(pie_data)]
                    wedges, texts, autotexts = ax.pie(
                        pie_data.values, 
                        labels=pie_data.index, 
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=colors,
                        textprops={'fontsize': 10}
                    )
                    # Enhance text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    # Add title with value column info if using aggregated data
                    if value_col:
                        title = f"{col} Distribution (by {value_col})"
                    else:
                        title = f"{col} Distribution"
                    
                    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
                    ax.axis("equal")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
    
    def _decide_chart_type_with_llm(self, user_query: str) -> str:
        """Use LLM to dynamically decide the best chart type based on the user's question."""
        prompt = f"""You are an expert data visualization assistant.
Based on the user's query and the data properties, decide the single most appropriate chart type to display.

USER QUERY: "{user_query}"

AVAILABLE DATA:
- Numeric columns: {', '.join(self.insights.numeric_cols) if self.insights.numeric_cols else 'None'}
- Categorical columns: {', '.join(self.insights.categorical_cols) if self.insights.categorical_cols else 'None'}
- Date columns: {', '.join(self.insights.date_cols) if self.insights.date_cols else 'None'}

CHART TYPE OPTIONS:
1. "bar" - Best for comparing numeric values across categories, or showing distributions of numeric data.
2. "pie" - Best for showing proportions or percentages of categorical data (only if categorical columns exist).
3. "trend" - Best for showing changes over time (only if both date and numeric columns exist).
4. "correlation" - Best for showing relationships between multiple numeric variables (only if 2+ numeric columns exist).
5. "auto" - If you are unsure or want to display all relevant charts.

Respond with ONLY ONE word from the options above: bar, pie, trend, correlation, or auto.
"""
        try:
            response = call_llm(prompt, temperature=0.1)
            if response:
                choice = response.strip().lower()
                for valid_choice in ["bar", "pie", "trend", "correlation", "auto"]:
                    if valid_choice in choice:
                        return valid_choice
            return "auto"
        except Exception as e:
            logger.warning("LLM chart detection failed: %s", e)
            return "auto"

    def render_insights(self, chart_type: str = "auto", user_query: str = None):
        """Main method to render all insights"""
        # st.markdown("### 📊 Dataset Insights & Analysis")
        # st.markdown("Comprehensive analysis of your query results:")
        
        # Executive summary
        self.render_executive_summary()
        
        st.divider()

        if chart_type == "auto" and user_query:
            llm_chart_type = self._decide_chart_type_with_llm(user_query)
            if llm_chart_type != "auto":
                st.info(f"🤖 selecting  '{llm_chart_type}' chart based on your query.")
                chart_type = llm_chart_type

        if chart_type == "bar":
            if self.insights.numeric_cols:
                self.render_numeric_analysis()
            elif self.insights.categorical_cols:
                st.info("No numeric columns found. Showing categorical pie charts instead.")
                self.render_categorical_analysis()
        elif chart_type == "pie":
            if self.insights.categorical_cols:
                self.render_categorical_analysis()
            elif self.insights.numeric_cols:
                st.info("No categorical columns found. Showing numeric bar charts instead.")
                self.render_numeric_analysis()
        elif chart_type == "trend":
            if self.insights.date_cols and self.insights.numeric_cols:
                self.render_trend_analysis()
            else:
                st.info("Missing date or numeric columns for trend analysis. Showing all relevant charts.")
                chart_type = "auto"
        elif chart_type == "correlation":
            if len(self.insights.numeric_cols) >= 2:
                self.render_correlation_analysis()
            else:
                st.info("Need at least two numeric columns for correlation analysis. Showing all relevant charts.")
                chart_type = "auto"

        if chart_type == "auto":
            if self.insights.numeric_cols:
                self.render_numeric_analysis()
                st.divider()
            if self.insights.categorical_cols:
                self.render_categorical_analysis()
                st.divider()
            if len(self.insights.numeric_cols) >= 2:
                self.render_correlation_analysis()
                st.divider()
            if self.insights.date_cols and self.insights.numeric_cols:
                self.render_trend_analysis()
                st.divider()




def generate_insights(result_df: pd.DataFrame, chart_type: str = "auto", user_query: str = None) -> None:
    """
    Main entry point for generating insights
    
    Args:
        result_df: DataFrame with query results
        chart_type: Chart type to render
        user_query: Original user query from the user to dynamically determine chart representations
    """
    if result_df is None or result_df.empty:
        return
    
    try:
        # Create insights generator and visualizer
        insights_gen = DataInsightsGenerator(result_df)
        visualizer = InsightVisualizer(result_df, insights_gen)
        
        # Render all insights
        visualizer.render_insights(chart_type=chart_type, user_query=user_query)
        
    except Exception as e:
        logger.exception(f"Error generating insights: {e}")
        st.error(f"Could not generate insights: {str(e)}")
