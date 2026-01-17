import plotly.express as px

plot_df = pd.DataFrame(normalized_embedding, columns=["x", "y"])
plot_df["GS_ID"] = list(prompts.keys())
# plot_df["NAME"] = [f'({key}) {pathway_info[key]["NAME"]}<br>{" ".join([p.split("), ")[-1] for p in prompts[key][1].split(" (")][:-1])}' for key in pathway_info.keys()]
plot_df["NAME"] = [
    f'({key}) {pathway_info[key]["NAME"]}' for key in pathway_info.keys()
]
plot_df["Color"] = colors
unique_labels = plot_df["Color"].unique()
color_map = {
    label: color for label, color in zip(unique_labels, px.colors.qualitative.Plotly)
}
plot_df["Color"] = plot_df["Color"].map(color_map)


fig = px.scatter(
    plot_df, x="x", y="y", hover_name="NAME", width=800, height=800, opacity=0.7
)  # color='Color'
fig.update_layout(showlegend=True)
fig.update_layout(
    title=f"t-SNE projection of llm2vec embeddings from {plot_df.shape[0]} pathways",
    xaxis_title="TSNE-1",
    yaxis_title="TSNE-2",
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12, color="Black"),
)
fig.show()


plot_df = pd.DataFrame(normalized_embedding, columns=["x", "y"])
plot_df["GS_ID"] = list(prompts.keys())
plot_df["NAME"] = [
    f'({key}) {pathway_info[key]["NAME"]}<br>{" ".join([p.split("), ")[-1] for p in prompts[key][1].split(" (")][:-1])}'
    for key in pathway_info.keys()
]

plot_df["Color"] = colors
unique_labels = plot_df["Color"].unique()
color_map = {
    label: color for label, color in zip(unique_labels, px.colors.qualitative.Plotly)
}
plot_df["Color"] = plot_df["Color"].map(color_map)


fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    hover_name="NAME",
    color="Color",
    width=800,
    height=800,
    opacity=0.7,
)  # color='Color'
fig.update_layout(showlegend=True)
fig.update_layout(
    title=f"t-SNE projection of llm2vec embeddings from {plot_df.shape[0]} pathways",
    xaxis_title="TSNE-1",
    yaxis_title="TSNE-2",
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12, color="Black"),
)
fig.show()
