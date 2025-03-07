"""
Transcript report visualization module.
Provides functionality for generating visual reports from transcript analysis.
"""
import os
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import io
import base64

def generate_topic_network(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a topic network visualization showing relationships 
    between topics and transcripts.
    
    Args:
        analysis_results: The analysis results
        
    Returns:
        Base64 encoded PNG image
    """
    # Create a graph
    G = nx.Graph()
    
    # Add transcript nodes
    for transcript in analysis_results.get('transcripts_metadata', []):
        transcript_id = transcript.get('id', '')
        title = transcript.get('title', 'Untitled')
        date = transcript.get('date', '').split('T')[0] if transcript.get('date') else ''
        G.add_node(transcript_id, type='transcript', label=f"{title} ({date})")
    
    # Add topic nodes and edges
    for topic in analysis_results.get('common_topics', []):
        topic_name = topic.get('name', 'Unknown')
        G.add_node(topic_name, type='topic', label=topic_name)
        
        # Connect topics to transcripts
        for occurrence in topic.get('transcripts', []):
            transcript_id = occurrence.get('id', '')
            if transcript_id in G:
                G.add_edge(topic_name, transcript_id)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G)
    
    # Draw transcript nodes
    transcript_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'transcript']
    nx.draw_networkx_nodes(G, pos, nodelist=transcript_nodes, 
                         node_color='lightblue', node_size=600, alpha=0.8)
    
    # Draw topic nodes
    topic_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'topic']
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, 
                         node_color='lightgreen', node_size=800, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    labels = {n: d['label'] if len(d['label']) < 20 else d['label'][:17]+'...' 
              for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title("Topic-Transcript Network")
    plt.axis('off')
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def generate_topic_evolution_chart(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a chart showing topic evolution over time.
    
    Args:
        analysis_results: The analysis results
        
    Returns:
        Base64 encoded PNG image
    """
    evolving_topics = analysis_results.get('evolving_topics', [])
    if not evolving_topics:
        return None
    
    # Collect data for chart
    topic_data = []
    for topic in evolving_topics:
        topic_name = topic.get('name', 'Unknown')
        for mention in topic.get('evolution', []):
            date = mention.get('date', '')
            if date:
                topic_data.append({
                    'topic': topic_name,
                    'date': datetime.strptime(date, '%Y-%m-%d'),
                    'mentioned': 1
                })
    
    if not topic_data:
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(topic_data)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create a pivot table: topics x dates
    pivot = df.pivot_table(
        index='topic', 
        columns='date', 
        values='mentioned',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create the chart
    plt.figure(figsize=(12, 6))
    
    # Plot each topic as a line
    for topic in pivot.index:
        plt.plot(pivot.columns, pivot.loc[topic], marker='o', label=topic)
    
    plt.title("Topic Evolution Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mentions")
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def generate_action_item_status_chart(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a chart showing action item status.
    
    Args:
        analysis_results: The analysis results
        
    Returns:
        Base64 encoded PNG image
    """
    action_items = analysis_results.get('action_item_status', [])
    if not action_items:
        return None
    
    # Count statuses
    status_counts = {
        'completed': 0,
        'in_progress': 0,
        'pending': 0,
        'at_risk': 0,
        'overdue': 0,
        'canceled': 0
    }
    
    for item in action_items:
        status = item.get('status', 'pending')
        if status in status_counts:
            status_counts[status] += 1
    
    # Create chart
    plt.figure(figsize=(8, 6))
    
    # Define colors
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#FF9800', '#F44336', '#9E9E9E']
    
    # Create pie chart
    labels = ['Completed', 'In Progress', 'Pending', 'At Risk', 'Overdue', 'Canceled']
    sizes = [status_counts['completed'], status_counts['in_progress'], 
             status_counts['pending'], status_counts['at_risk'],
             status_counts['overdue'], status_counts['canceled']]
    
    # Skip empty values
    non_zero_labels = []
    non_zero_sizes = []
    non_zero_colors = []
    
    for i, size in enumerate(sizes):
        if size > 0:
            non_zero_labels.append(labels[i])
            non_zero_sizes.append(size)
            non_zero_colors.append(colors[i])
    
    if not non_zero_sizes:
        return None
        
    plt.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title("Action Item Status")
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def generate_html_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate an HTML report with visualizations.
    
    Args:
        analysis_results: The analysis results
        
    Returns:
        HTML report as string
    """
    # Generate visualizations
    topic_network = generate_topic_network(analysis_results)
    topic_evolution = generate_topic_evolution_chart(analysis_results)
    action_item_status = generate_action_item_status_chart(analysis_results)
    
    # Start building HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Meeting Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #333; }
            .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .visualization { margin: 30px 0; text-align: center; }
            .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .high-priority { background-color: #ffebee; }
            .status-completed { color: #4CAF50; }
            .status-in-progress { color: #2196F3; }
            .status-pending { color: #FFC107; }
            .status-at-risk { color: #FF9800; }
            .status-overdue { color: #F44336; }
        </style>
    </head>
    <body>
        <h1>Meeting Analysis Report</h1>
    """
    
    # Add metadata
    transcripts = analysis_results.get('transcripts_metadata', [])
    if transcripts:
        html += """
        <div class="metadata">
            <p><strong>Analysis Date:</strong> {0}</p>
            <p><strong>Number of Meetings:</strong> {1}</p>
        </div>
        """.format(
            datetime.now().strftime("%Y-%m-%d"),
            len(transcripts)
        )
    
    # Add summary
    html += """
        <div class="summary">
            <h2>Executive Summary</h2>
            <div>{0}</div>
        </div>
    """.format(analysis_results.get('comparative_summary', 'No summary available.').replace('\n', '<br>'))
    
    # Add visualizations
    if topic_network:
        html += """
        <div class="visualization">
            <h2>Topic Network</h2>
            <img src="data:image/png;base64,{0}" alt="Topic Network">
            <p class="caption">This visualization shows how topics are connected across meetings.</p>
        </div>
        """.format(topic_network)
    
    if topic_evolution:
        html += """
        <div class="visualization">
            <h2>Topic Evolution Over Time</h2>
            <img src="data:image/png;base64,{0}" alt="Topic Evolution">
            <p class="caption">This chart shows how topics have evolved across meetings over time.</p>
        </div>
        """.format(topic_evolution)
    
    if action_item_status:
        html += """
        <div class="visualization">
            <h2>Action Item Status</h2>
            <img src="data:image/png;base64,{0}" alt="Action Item Status">
            <p class="caption">This chart shows the status breakdown of all action items.</p>
        </div>
        """.format(action_item_status)
    
    # Add common topics table
    html += """
        <h2>Key Topics Across Meetings</h2>
        <table>
            <tr>
                <th>Topic</th>
                <th>Frequency</th>
                <th>Business Impact</th>
                <th>Description</th>
            </tr>
    """
    
    for topic in analysis_results.get('common_topics', []):
        # Check if business_priority exists, otherwise use a default
        impact = topic.get('business_priority', topic.get('business_impact', 'Medium'))
        
        html += """
            <tr class="{3}-priority">
                <td>{0}</td>
                <td>{1}</td>
                <td>{2}</td>
                <td>{3}</td>
            </tr>
        """.format(
            topic.get('name', 'Unknown'),
            topic.get('frequency', 0),
            impact,
            topic.get('description', '')
        )
    
    html += """
        </table>
    """
    
    # Add conflicts/changes table if available
    conflicts = analysis_results.get('conflicting_information', [])
    if conflicts:
        html += """
            <h2>Conflicts & Changes</h2>
            <table>
                <tr>
                    <th>Topic</th>
                    <th>Risk Level</th>
                    <th>Recommendation</th>
                    <th>Details</th>
                </tr>
        """
        
        for conflict in conflicts:
            changes_html = "<ul>"
            for change in conflict.get('changes', []):
                changes_html += "<li><strong>{0}:</strong> {1}</li>".format(
                    change.get('date', 'Unknown date'),
                    change.get('description', 'No description')
                )
            changes_html += "</ul>"
            
            html += """
                <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                    <td>{2}</td>
                    <td>{3}</td>
                </tr>
            """.format(
                conflict.get('topic', 'Unknown'),
                conflict.get('risk_level', 'Medium'),
                conflict.get('recommendation', 'No recommendation provided'),
                changes_html
            )
        
        html += """
            </table>
        """
    
    # Add action items table
    html += """
        <h2>Action Items</h2>
        <table>
            <tr>
                <th>Description</th>
                <th>Assignee</th>
                <th>Status</th>
                <th>Priority</th>
                <th>First Mentioned</th>
            </tr>
    """
    
    for item in analysis_results.get('action_item_status', []):
        # Determine CSS class for status
        status = item.get('status', 'pending')
        status_class = f"status-{status}" if status in ['completed', 'in_progress', 'pending', 'at_risk', 'overdue'] else ""
        
        # Determine CSS class for priority
        priority = item.get('priority', 'Medium')
        priority_class = "high-priority" if priority == "High" else ""
        
        html += """
            <tr class="{5}">
                <td>{0}</td>
                <td>{1}</td>
                <td class="{2}">{3}</td>
                <td>{4}</td>
                <td>{6}</td>
            </tr>
        """.format(
            item.get('description', ''),
            item.get('assignee', 'Unassigned'),
            status_class,
            status.capitalize(),
            priority,
            priority_class,
            item.get('first_mentioned', '')
        )
    
    html += """
        </table>
    """
    
    # Add meeting details
    html += """
        <h2>Analyzed Meetings</h2>
        <table>
            <tr>
                <th>Title</th>
                <th>Date</th>
                <th>Topic</th>
            </tr>
    """
    
    for transcript in sorted(transcripts, key=lambda x: x.get('date', '0000-00-00')):
        html += """
            <tr>
                <td>{0}</td>
                <td>{1}</td>
                <td>{2}</td>
            </tr>
        """.format(
            transcript.get('title', 'Untitled'),
            transcript.get('date', '').split('T')[0] if transcript.get('date') else 'Unknown date',
            transcript.get('topic', 'No topic specified')
        )
    
    html += """
        </table>
    """
    
    # Close HTML
    html += """
    <div class="footer">
        <p>Generated on {0}</p>
    </div>
    </body>
    </html>
    """.format(datetime.now().strftime("%Y-%m-%d at %H:%M"))
    
    return html

def save_html_report(analysis_results: Dict[str, Any], output_path: str) -> None:
    """
    Generate and save an HTML report.
    
    Args:
        analysis_results: The analysis results
        output_path: Path to save the HTML report
        
    Returns:
        None
    """
    html = generate_html_report(analysis_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")