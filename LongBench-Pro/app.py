#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongBenchmark Results Visualization
"""

import json
import re
import pandas as pd
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go

with open('./output/model_info.json', 'r', encoding='utf-8') as f:
    MODLE_INFO_DICT = json.load(f)

def get_color(index):
    """Generate color based on index, using golden angle to ensure uniform and infinite color distribution"""
    # Golden angle approx 137.508 degrees
    hue = (index * 137.508) % 360
    # Fixed saturation 70%, lightness 60%
    return f"hsl({hue}, 70%, 60%)"

# Custom CSS
CUSTOM_CSS = """
/* Force title center */
h1 {
    text-align: center;
    display: block;
}

/* Header center */
#leaderboard_table th, 
#leaderboard_table th button, 
#leaderboard_table th span {
    text-align: center !important;
    justify-content: center !important;
}

/* Content column center: starting from 3rd column */
#leaderboard_table td:nth-child(n+3) {
    text-align: center !important;
}

/* Make tab labels bold */
button[role="tab"] {
    font-weight: bold !important;
}
"""

class ResultParser:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.results = []
        
    def parse_filename(self, filename: str):
        """Parse filename to extract context length and thinking status"""
        # Extract context length
        context_match = re.search(r'context-(\d+)', filename)
        context_length = int(context_match.group(1)) if context_match else 0
        
        filename_lower = filename.lower()
        # Check nonthinking
        has_nonthinking = 'nonthinking' in filename_lower
        # Check thinking
        has_thinking = 'thinking' in filename_lower and not has_nonthinking
        
        return context_length, has_thinking, has_nonthinking
    
    def parse_result_file(self, model_name: str, file_path: Path):
        """Parse single result file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            context_length, has_thinking, has_nonthinking = self.parse_filename(file_path.name)
            # Use date field as evaluation date
            eval_date = data.get('date', "Unknown")
            
            # Extract BoN data
            bon_data = {}
            for bon_key in ['BoN-1', 'BoN-2', 'BoN-3']:
                if bon_key in data and 'overall_metric' in data[bon_key]:
                    bon_data[bon_key] = data[bon_key]['overall_metric']
            
            result = {
                'model_name': model_name,
                'eval_date': eval_date,
                'context_length': context_length,
                'has_thinking': has_thinking,
                'has_nonthinking': has_nonthinking,
                'overall_metric': data.get('average_overall_metric', 0.0),
                'token_length_metrics': data.get('average_token_length_metric', {}),
                'contextual_requirement': data.get('average_contextual_requirement_metric', {}),
                'difficulty': data.get('average_difficulty_metric', {}),
                'primary_task': data.get('average_primary_task_metric', {}),
                'language': data.get('average_language_metric', {}),
                'bon_data': bon_data,  # Store BoN-1, BoN-2, BoN-3 overall_metric
                'pass_at_k': {
                    'Pass@1': data.get('pass@1'),
                    'Pass@2': data.get('pass@2'),
                    'Pass@3': data.get('pass@3')
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None
    
    def scan_all_results(self):
        """Scan all model result files"""
        self.results = []
        
        if not self.output_dir.exists():
            print(f"Output directory does not exist: {self.output_dir}")
            return
        
        # Traverse all model directories
        for model_dir in self.output_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            print(f"Scanning model: {model_name}")
            
            # Find all _summary.json files
            for file_path in model_dir.glob("*_summary.json"):
                print(f"  Parsing file: {file_path.name}")
                result = self.parse_result_file(model_name, file_path)
                if result:
                    self.results.append(result)
        
        print(f"Total parsed {len(self.results)} result files")
    
    def get_leaderboard_data(self):
        """Get leaderboard data"""
        if not self.results:
            return pd.DataFrame()
        
        # Aggregate data by model name
        model_groups = {}
        for result in self.results:
            model_name = result['model_name']
            if model_name not in model_groups:
                model_groups[model_name] = {
                    'dates': [],
                    'contexts': [],
                    'thinking_scores': [],
                    'non_thinking_scores': []
                }
            
            group = model_groups[model_name]
            group['dates'].append(result['eval_date'])
            group['contexts'].append(result['context_length'])
            
            score = result['overall_metric']
            if result['has_thinking']:
                group['thinking_scores'].append(score)
            else:
                group['non_thinking_scores'].append(score)
        
        leaderboard_data = []
        for model_name, group in model_groups.items():
            # Get latest date
            valid_dates = [d for d in group['dates'] if d != "Unknown"]
            latest_date = max(valid_dates) if valid_dates else "Unknown"
            
            # Get max Context Window
            max_context = max(group['contexts']) if group['contexts'] else 0
            
            # Format truncated length
            if max_context >= 1000000:
                context_str = f"{max_context/1000000:.0f}M" if max_context % 1000000 == 0 else f"{max_context/1000000:.1f}M"
            elif max_context >= 1000:
                context_str = f"{max_context/1000:.0f}k" if max_context % 1000 == 0 else f"{max_context/1000:.1f}k"
            else:
                context_str = str(max_context)
            
            # Get model type and context length
            model_context = "-"
            model_url = ""
            if model_name in MODLE_INFO_DICT:
                model_info = MODLE_INFO_DICT[model_name]
                if isinstance(model_info, dict):
                    model_type = model_info.get("type", "Unknown")
                    model_context = model_info.get("context_length", "-")
                    model_url = model_info.get("url", "")
                else:
                    model_type = str(model_info)
            else:
                model_type = "Unknown"
            
            # Handle model name link and icon
            display_model_name = model_name

            if model_url:
                display_model_name = f"[{display_model_name}]({model_url})"

            # Calculate average score
            nt_score_val = 0
            nt_score_str = "-"
            if group['non_thinking_scores']:
                nt_score_val = sum(group['non_thinking_scores']) / len(group['non_thinking_scores'])
                nt_score_str = f"{nt_score_val * 100:.2f}"
                
            t_score_val = 0
            t_score_str = "-"
            if group['thinking_scores']:
                t_score_val = sum(group['thinking_scores']) / len(group['thinking_scores'])
                t_score_str = f"{t_score_val * 100:.2f}"
            
            leaderboard_data.append({
                'Model Name': display_model_name,
                'Model Type': model_type,
                'Context Length': model_context,
                'Truncated Length': context_str,
                'Non-Thinking Score': nt_score_str,
                'Thinking Score': t_score_str,
                '_sort_score': max(nt_score_val, t_score_val)
            })
        
        df = pd.DataFrame(leaderboard_data)
        # Sort by highest score descending
        if not df.empty:
            df = df.sort_values('_sort_score', ascending=False).drop(columns=['_sort_score']).reset_index(drop=True)
            
        return df

def get_display_name_for_result(result):
    """Get display name for model (append suffix based on thinking/nonthinking)"""
    if result.get('has_nonthinking'):
        return f"{result['model_name']}_nonthinking"
    elif result.get('has_thinking'):
        return f"{result['model_name']}_thinking"
    else:
        return result['model_name']

def get_model_color_index(model_name, all_models):
    """Get model index in color list"""
    try:
        return all_models.index(model_name)
    except ValueError:
        return 0

def create_contextual_requirement_chart(results, selected_models):
    """Create contextual requirement comparison bar chart"""
    if not selected_models:
        return go.Figure()
    
    # Collect data
    chart_data = {}
    
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            model_name = display_name
            contextual_requirement = result['contextual_requirement']
            
            # Store each model's result directly
            if model_name not in chart_data:
                chart_data[model_name] = {}
            
            for req_type, score in contextual_requirement.items():
                chart_data[model_name][req_type] = score * 100  # multiply by 100
    
    # Create chart
    fig = go.Figure()
    
    # Get all requirement types
    all_req_types = []
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            contextual_requirement = result['contextual_requirement']
            for req_type in contextual_requirement.keys():
                if req_type not in all_req_types:
                    all_req_types.append(req_type)
    
    for model_name in selected_models:
        if model_name in chart_data:
            scores = [chart_data[model_name].get(req_type, 0) for req_type in all_req_types]
            color_index = get_model_color_index(model_name, selected_models)
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=all_req_types,
                y=scores,
                marker_color=get_color(color_index),
                text=[f"{score:.2f}" for score in scores],  # keep 2 decimal places
                textposition='auto'
            ))
    
    fig.update_layout(
        title='Performance Comparison on Different Context Requirements',
        xaxis_title='Context Requirement Type',
        yaxis_title='Average Score',
        barmode='group',
        autosize=True,  # auto size
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,  # adjust lower
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)  # increase bottom margin
    )
    
    return fig

def create_primary_task_radar_chart(results, selected_models):
    """Create primary task radar chart (aggregate by prefix)"""
    if not selected_models:
        return go.Figure()
    
    # Collect all model task prefixes
    prefix_order = []
    # Map prefix -> [scores] for each model
    model_prefix_scores = {}
    
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name not in selected_models:
            continue
        primary_task = result.get('primary_task', {})
        if display_name not in model_prefix_scores:
            model_prefix_scores[display_name] = {}
        for task_key, score in primary_task.items():
            prefix = task_key.split('.')[0].strip() if isinstance(task_key, str) else str(task_key)
            if prefix not in prefix_order:
                prefix_order.append(prefix)
            if prefix not in model_prefix_scores[display_name]:
                model_prefix_scores[display_name][prefix] = []
            model_prefix_scores[display_name][prefix].append(score * 100)
    
    # Take first 11 prefixes
    categories = prefix_order[:11]
    
    # Create radar chart
    fig = go.Figure()
    
    for model_name in selected_models:
        if model_name not in model_prefix_scores:
            continue
        # Mean aggregation for each prefix
        values = []
        for prefix in categories:
            scores = model_prefix_scores[model_name].get(prefix, [])
            if scores:
                values.append(sum(scores) / len(scores))
            else:
                values.append(0)
        # Close polygon
        r_values = values + ([values[0]] if values else [])
        theta_values = categories + ([categories[0]] if categories else [])
        color_index = get_model_color_index(model_name, selected_models)
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            mode='lines+markers',
            name=model_name,
            line=dict(color=get_color(color_index), width=3),
            marker=dict(size=6),
            fill='toself'
        ))
    
    fig.update_layout(
        title='Performance Comparison on Different Primary Tasks',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )
    
    return fig

def create_language_chart(results, selected_models):
    """Create language comparison bar chart"""
    if not selected_models:
        return go.Figure()
    
    # Collect data
    chart_data = {}
    
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            model_name = display_name
            language = result['language']
            
            # Store each model's result directly
            if model_name not in chart_data:
                chart_data[model_name] = {}
            
            for lang_type, score in language.items():
                chart_data[model_name][lang_type] = score * 100  # multiply by 100
    
    # Create chart
    fig = go.Figure()
    
    # Get all language types
    all_lang_types = []
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            language = result['language']
            for lang_type in language.keys():
                if lang_type not in all_lang_types:
                    all_lang_types.append(lang_type)
    
    for model_name in selected_models:
        if model_name in chart_data:
            scores = [chart_data[model_name].get(lang_type, 0) for lang_type in all_lang_types]
            color_index = get_model_color_index(model_name, selected_models)
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=all_lang_types,
                y=scores,
                marker_color=get_color(color_index),
                text=[f"{score:.2f}" for score in scores],  # keep 2 decimal places
                textposition='auto'
            ))
    
    fig.update_layout(
        title='Performance Comparison on Different Languages',
        xaxis_title='Language Type',
        yaxis_title='Average Score',
        barmode='group',
        autosize=True,  # auto size
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,  # adjust lower
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)  # increase bottom margin
    )
    
    return fig

def create_difficulty_chart(results, selected_models):
    """Create difficulty comparison bar chart"""
    if not selected_models:
        return go.Figure()
    
    # Collect data
    chart_data = {}
    
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            model_name = display_name
            difficulty = result['difficulty']
            
            # Store each model's result directly
            if model_name not in chart_data:
                chart_data[model_name] = {}
            
            for diff_type, score in difficulty.items():
                chart_data[model_name][diff_type] = score * 100  # multiply by 100
    
    # Create chart
    fig = go.Figure()
    
    # Get all difficulty types
    all_diff_types = []
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            difficulty = result['difficulty']
            for diff_type in difficulty.keys():
                if diff_type not in all_diff_types:
                    all_diff_types.append(diff_type)
    
    for model_name in selected_models:
        if model_name in chart_data:
            scores = [chart_data[model_name].get(diff_type, 0) for diff_type in all_diff_types]
            color_index = get_model_color_index(model_name, selected_models)
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=all_diff_types,
                y=scores,
                marker_color=get_color(color_index),
                text=[f"{score:.2f}" for score in scores],  # keep 2 decimal places
                textposition='auto'
            ))
    
    fig.update_layout(
        title='Performance Comparison on Different Difficulties',
        xaxis_title='Difficulty Type',
        yaxis_title='Average Score',
        barmode='group',
        autosize=True,  # auto size
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,  # adjust lower
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)  # increase bottom margin
    )
    
    return fig

def create_length_heatmap(results, selected_models):
    """Create length heatmap"""
    if not selected_models:
        return go.Figure()
    
    # Standard context lengths
    standard_lengths = [8000, 16000, 32000, 64000, 128000, 256000]
    standard_length_keys = ['8k', '16k', '32k', '64k', '128k', '256k']
    
    # Map results by name
    result_map = {get_display_name_for_result(r): r for r in results}
    
    # Prepare heatmap data
    heatmap_data = []
    model_names = []
    
    for model_name in selected_models:
        if model_name in result_map:
            model_names.append(model_name)
            result = result_map[model_name]
            
            # Get data from token_length_metrics
            token_length_metrics = result.get('token_length_metrics', {})
            row_data = []
            
            for key in standard_length_keys:
                if key in token_length_metrics:
                    row_data.append(token_length_metrics[key] * 100)  # multiply by 100
                else:
                    row_data.append(None)  # No data point
            
            heatmap_data.append(row_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{length//1000}k" for length in standard_lengths],  # x axis labels
        y=model_names,  # y axis labels
        colorscale='RdYlBu_r',  # Red is low, Blue is high
        showscale=True,
        text=[[f"{val:.2f}" if val is not None else "N/A" for val in row] for row in heatmap_data],  # show values
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Performance Heatmap on Different Sample Lengths',
        xaxis_title='Sample Length (tokens)',
        yaxis_title='Model Name',
        autosize=True,
        height=max(400, len(model_names) * 50),  # adjust height based on model count
        margin=dict(l=150, r=50, t=80, b=80)  # adjust margins
    )
    
    return fig

def create_bon_chart(results, selected_models):
    """Create BoN 1-3 line chart"""
    if not selected_models:
        return go.Figure()
    
    # BoN labels
    bon_labels = ['BoN-1', 'BoN-2', 'BoN-3']
    bon_indices = [1, 2, 3]
    
    # Prepare data for each model
    model_data = {}
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            if display_name not in model_data:
                model_data[display_name] = {}
            
            # Get data from bon_data
            bon_data = result.get('bon_data', {})
            for bon_key in bon_labels:
                if bon_key in bon_data:
                    bon_index = bon_labels.index(bon_key) + 1
                    model_data[display_name][bon_index] = bon_data[bon_key] * 100  # multiply by 100
    
    # Create chart
    fig = go.Figure()
    
    for model_name in selected_models:
        if model_name not in model_data:
            continue
            
        data = model_data[model_name]
        if not data:
            continue
        
        # Prepare data for each BoN
        x_values = []
        y_values = []
        text_values = []
        
        for bon_index in bon_indices:
            x_values.append(bon_index)
            if bon_index in data:
                y_values.append(data[bon_index])
                text_values.append(f"{data[bon_index]:.2f}")
            else:
                y_values.append(None)
                text_values.append("")
        
        # Get model color index
        color_index = get_model_color_index(model_name, selected_models)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=model_name,
            line=dict(color=get_color(color_index), width=3),
            marker=dict(size=10),
            text=text_values,
            textposition='top center',
            connectgaps=False
        ))
    
    # Set x axis
    fig.update_layout(
        title='Performance Comparison on Different Best-of-N',
        xaxis_title='N',
        yaxis_title='Average Score',
        autosize=True,
        xaxis=dict(
            tickmode='array',
            tickvals=bon_indices,
            ticktext=bon_labels,
            tickangle=0
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )
    
    return fig

def create_pass_k_chart(results, selected_models):
    """Create Pass@N line chart"""
    if not selected_models:
        return go.Figure()
    
    # Pass@K labels
    k_labels = ['Pass@1', 'Pass@2', 'Pass@3']
    k_indices = [1, 2, 3]
    
    # Prepare data for each model
    model_data = {}
    for result in results:
        display_name = get_display_name_for_result(result)
        if display_name in selected_models:
            if display_name not in model_data:
                model_data[display_name] = {}
            
            # Get data from pass_at_k
            pass_data = result.get('pass_at_k', {})
            for i, k_key in enumerate(k_labels):
                val = pass_data.get(k_key)
                if val is not None:
                    k_index = k_indices[i]
                    model_data[display_name][k_index] = val * 100  # multiply by 100
    
    # Create chart
    fig = go.Figure()
    
    for model_name in selected_models:
        if model_name not in model_data:
            continue
            
        data = model_data[model_name]
        if not data:
            continue
        
        # Prepare data for each Pass@K
        x_values = []
        y_values = []
        text_values = []
        
        for k_index in k_indices:
            x_values.append(k_index)
            if k_index in data:
                y_values.append(data[k_index])
                text_values.append(f"{data[k_index]:.2f}")
            else:
                y_values.append(None)
                text_values.append("")
        
        # Get model color index
        color_index = get_model_color_index(model_name, selected_models)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=model_name,
            line=dict(color=get_color(color_index), width=3),
            marker=dict(size=10),
            text=text_values,
            textposition='top center',
            connectgaps=False
        ))
    
    # Set x axis
    fig.update_layout(
        title='Performance Comparison on Different Pass@N',
        xaxis_title='N',
        yaxis_title='Pass@N (%)',
        autosize=True,
        xaxis=dict(
            tickmode='array',
            tickvals=k_indices,
            ticktext=k_labels,
            tickangle=0
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )
    
    return fig

def create_gradio_interface(parser: ResultParser):
    """Create Gradio interface"""
    
    def refresh_data():
        """Refresh data"""
        parser.scan_all_results()
        return parser.get_leaderboard_data()
    
    def get_model_choices():
        """Get model choices (distinguish by suffix for thinking/nonthinking)"""
        if not parser.results:
            return []
        display_names = set()
        for r in parser.results:
            name = get_display_name_for_result(r)
            display_names.add(name)
        models = sorted(list(display_names))
        return models
    
    def update_charts(selected_models):
        """Update all charts"""
        if not selected_models:
            return None, None, None, None, None, None, None
        
        length_heatmap = create_length_heatmap(parser.results, selected_models)
        contextual_chart = create_contextual_requirement_chart(parser.results, selected_models)
        primary_task_radar_chart = create_primary_task_radar_chart(parser.results, selected_models)
        language_chart = create_language_chart(parser.results, selected_models)
        difficulty_chart = create_difficulty_chart(parser.results, selected_models)
        bon_chart = create_bon_chart(parser.results, selected_models)
        pass_k_chart = create_pass_k_chart(parser.results, selected_models)
        
        return length_heatmap, contextual_chart, primary_task_radar_chart, language_chart, difficulty_chart, bon_chart, pass_k_chart
    
    # Create interface
    with gr.Blocks(title="LongBench Pro Results Visualization", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown("# LongBench Pro Results Visualization")

        gr.HTML("""
        <div style="text-align: center; display: flex; justify-content: center; gap: 10px; margin-bottom: 20px;">
            <a href="https://huggingface.co/datasets/caskcsg/LongBench-Pro" target="_blank"><img src="https://img.shields.io/badge/HF-Dataset-yellow?logo=huggingface&logoColor=white" alt="HF Dataset"></a>
            <a href="https://github.com/caskcsg/longcontext/tree/main/LongBench-Pro" target="_blank"><img src="https://img.shields.io/badge/Github-Code-blue?logo=github&logoColor=white" alt="Github Code"></a>
            <a href="https://huggingface.co/spaces/caskcsg/LongBench-Pro-Leaderboard" target="_blank"><img src="https://img.shields.io/badge/🏆-Leaderboard-red" alt="Leaderboard"></a>
            <a href="#" target="_blank"><img src="https://img.shields.io/badge/📄-Arxiv_Paper-green" alt="Paper"></a>
        </div>
        """)
        
        # Leaderboard area
        gr.Markdown("## 🏆 Overall Performance Leaderboard")
        gr.Markdown("""
        - *Thinking scores for Thinking and Mixed-Thinking models use their own thinking capabilities (Non-Thinking Prompt)*
        - *Thinking scores for Instruct models are obtained using thinking prompts (Thinking Prompt)*
        """)
        leaderboard_df = gr.Dataframe(
            headers=["Model Name", "Model Type", "Context Length", "Truncation Length", "Non-Thinking Score", "Thinking Score"],
            datatype=["markdown", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            show_row_numbers=True,
            show_search="filter",
            max_height=800,
            column_widths=["250px", "100px", "100px", "100px", "120px", "120px"],
            elem_id="leaderboard_table"
        )
        
        # Model selection and chart area
        gr.HTML("<br>")
        gr.Markdown("## 📊 Specific Dimension Comparison")
        with gr.Row():
            with gr.Column(scale=4):
                model_selector = gr.Dropdown(
                    choices=[],
                    label="Select Models",
                    value=[],
                    multiselect=True,
                    interactive=True
                )
            with gr.Column(scale=1):
                update_charts_btn = gr.Button("Update Charts", variant="primary", size="lg")
        
        with gr.Tabs():
            with gr.TabItem("Language"):
                language_plot = gr.Plot(show_label=False)
            
            with gr.TabItem("Difficulty"):
                difficulty_plot = gr.Plot(show_label=False)
            
            with gr.TabItem("Sample Length"):
                length_heatmap = gr.Plot(show_label=False)
            
            with gr.TabItem("Primary Task"):
                primary_task_radar_plot = gr.Plot(show_label=False)
            
            with gr.TabItem("Context Requirement"):
                contextual_plot = gr.Plot(show_label=False)
            
            with gr.TabItem("Best-of-N"):
                bon_plot = gr.Plot(show_label=False)
            
            with gr.TabItem("Pass@N"):
                pass_k_plot = gr.Plot(show_label=False)
            
        # Add bottom spacer
        gr.HTML("<div style='height: 100px;'></div>")
        
        # Event handling
        def update_model_choices():
            models = get_model_choices()
            return gr.Dropdown(choices=models, value=[])
        
        update_charts_btn.click(
            fn=update_charts,
            inputs=[model_selector],
            outputs=[length_heatmap, contextual_plot, primary_task_radar_plot, language_plot, difficulty_plot, bon_plot, pass_k_plot]
        )
        
        # Initialize
        demo.load(
            fn=refresh_data,
            outputs=[leaderboard_df]
        ).then(
            fn=update_model_choices,
            outputs=[model_selector]
        )
    
    return demo

def main():
    """Main function"""
    output_dir = "./output"
    
    print("Initializing result parser...")
    parser = ResultParser(output_dir)
    
    print("Scanning result files...")
    parser.scan_all_results()
    
    print("Creating Gradio interface...")
    demo = create_gradio_interface(parser)
    
    print("Starting server...")
    demo.launch()

if __name__ == "__main__":
    main()
