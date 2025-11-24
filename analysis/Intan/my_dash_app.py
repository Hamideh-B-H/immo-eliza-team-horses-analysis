import dash
import webbrowser
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from data_utils import DataStats

class AppManager():
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data_stats = DataStats()
        
    def price_per_province(self, df, provinces_sorted):
        """Creates a boxplot of price distribution by province."""
        # Create boxplot for price distribution by province
        fig = px.box(
            df,
            x="province",
            y="price",
            color="province",
            category_orders={"province": provinces_sorted},  # Sort alphabetically
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Update hover with count, mean, median
        for trace in fig.data:
            province = trace.name
            sub_df = df[df["province"] == province]["price"]
            count = sub_df.count()
            mean = sub_df.mean()
            median = sub_df.median()
            trace.hovertemplate = (
                f"<b>{province}</b><br>"
                "Price: %{y}<br>"
                f"Count: {count}<br>"
                f"Mean: {mean:.2f}<br>"
                f"Median: {median:.2f}<extra></extra>"
            )
        
        fig.update_layout(
            title="Price Distribution by Province",
            xaxis_title="Province",
            yaxis_title="Price",
            showlegend=True,
            legend_title_text="Province"
        )
        return fig

    def top5_price_range(self, df):
        """Creates a boxplot of the top 5 provinces with the biggest price range."""
        stats = df.groupby("province")["price"].agg(['min', 'max', 'mean', 'median'])
        stats['range'] = stats['max'] - stats['min']
        
        top5_provinces = stats.sort_values('range', ascending=False).head(5).index.tolist()
        
        filtered_df = df[df["province"].isin(top5_provinces)]
        
        fig = px.box(
            filtered_df,
            x="province",
            y="price",
            points="all",  # Show individual data points
            color="province",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            category_orders={"province": top5_provinces}
        )
        
        for trace in fig.data:
            province = trace.name
            sub_df = filtered_df[filtered_df["province"] == province]["price"]
            trace.hovertemplate = (
                f"<b>{province}</b><br>"
                "Price: %{y}<br>"
                f"Count: {sub_df.count()}<br>"
                f"Mean: {sub_df.mean():.2f}<br>"
                f"Median: {sub_df.median():.2f}<br>"
                f"Min: {sub_df.min():.2f}<br>"
                f"Max: {sub_df.max():.2f}<extra></extra>"
            )
        
        fig.update_layout(
            title="Top 5 Provinces by Price Range",
            boxmode='group',
            xaxis_title="Province",
            yaxis_title="Price",
            showlegend=True
        )
        return fig

    def correlation_categorical_continuous(self, df, cat_var, cont_var, exclude_categories=None):
        """Creates a boxplot showing correlation between a categorical and a continuous variable."""
        if exclude_categories:
            df = df[~df[cat_var].isin(exclude_categories)]
        
        group_stats = df.groupby(cat_var)[cont_var].agg(['count', 'mean'])
        overall_mean = df[cont_var].mean()
        ss_between = sum(group_stats['count'] * (group_stats['mean'] - overall_mean)**2)
        ss_total = sum((df[cont_var] - overall_mean)**2)
        eta_squared = ss_between / ss_total
        
        categories_sorted = sorted(df[cat_var].unique())
        
        fig = px.box(
            df,
            x=cat_var,
            y=cont_var,
            points="all",
            color=cat_var,
            category_orders={cat_var: categories_sorted},
            color_discrete_sequence=px.colors.qualitative.Pastel
        ).update_layout(
            title=f"{cont_var} by {cat_var} (η² = {eta_squared:.3f})",
            xaxis_title=cat_var,
            yaxis_title=cont_var,
            boxmode='group'
        )
        
        return fig

    def plot_predictor_coeff(self, df):
        """Creates a plot showing the predictor coefficients."""
        coef_df = self.data_stats.find_predictor_coeff(df)
        
        fig = go.Figure(go.Bar(
            x=coef_df["abs_coef"],
            y=coef_df["feature"],
            orientation='h',
            marker=dict(color=coef_df["abs_coef"], colorscale='Viridis')
        ))

        fig.update_layout(
            title="Predictor Coefficients (Lasso Regression)",
            xaxis_title="Coefficient Size (Importance)",
            yaxis_title="Feature",
            template="plotly_white"
        )
        
        return fig

    def run(self, df, provinces_sorted):
        """Runs the Dash app with tabs for switching between plots."""
        
        self.app.layout = html.Div([
            html.H1("Real Estate Data Analysis"),
            
            dcc.Tabs(id="tabs", value='tab-1', children=[
                dcc.Tab(label='Price Distribution by Province', value='tab-1'),
                dcc.Tab(label='Top 5 Price Range by Province', value='tab-2'),
                dcc.Tab(label='Correlation between Categorical and Continuous', value='tab-3'),
                dcc.Tab(label='Predictor Coefficients', value='tab-4'),
            ]),
            html.Div(id='tabs-content'),
        ])
        
        @self.app.callback(
            Output('tabs-content', 'children'),
            Input('tabs', 'value')
        )
        def render_content(tab):
            if tab == 'tab-1':
                return dcc.Graph(figure=self.price_per_province(df, provinces_sorted))
            elif tab == 'tab-2':
                return dcc.Graph(figure=self.top5_price_range(df))
            elif tab == 'tab-3':
                return dcc.Graph(figure=self.correlation_categorical_continuous(df, "property_type", "price"))
            elif tab == 'tab-4':
                return dcc.Graph(figure=self.plot_predictor_coeff(df))

        # Open Dash app in Firefox explicitly
        webbrowser.get('firefox').open('http://127.0.0.1:8050/')
        
        # Run the app (with reloader set to False to avoid Jupyter Notebook conflict)
        self.app.run_server(debug=True, use_reloader=False)