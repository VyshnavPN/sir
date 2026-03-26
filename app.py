import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Misinformation Spread on Social Media", layout="wide")

st.title("📢 Misinformation Spread on Social Media")
st.markdown("""
### Model how false information spreads through a social network
This project combines:
- **SIR-style epidemic model**
- **Network structure**
- **Centrality effects**
- **Threshold-based adoption**

### States:
- **S = Skeptics**
- **I = Believers**
- **R = Fact-checkers**
""")

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙ Simulation Settings")

num_nodes = st.sidebar.slider("Number of Users (Nodes)", 20, 200, 60)
edge_prob = st.sidebar.slider("Connection Probability", 0.01, 0.30, 0.08, 0.01)
initial_believers = st.sidebar.slider("Initial Believers", 1, 20, 3)
initial_factcheckers = st.sidebar.slider("Initial Fact-checkers", 1, 20, 2)

infection_prob = st.sidebar.slider("Spread Probability (S → I)", 0.0, 1.0, 0.35, 0.01)
recovery_prob = st.sidebar.slider("Fact-check Conversion (I → R)", 0.0, 1.0, 0.20, 0.01)
skeptic_to_fact_prob = st.sidebar.slider("Skeptic to Fact-checker Probability (S → R)", 0.0, 1.0, 0.05, 0.01)

threshold_fraction = st.sidebar.slider("Threshold Fraction for Belief Adoption", 0.0, 1.0, 0.30, 0.01)
centrality_boost = st.sidebar.slider("Centrality Influence Boost", 0.0, 2.0, 0.8, 0.1)

steps = st.sidebar.slider("Simulation Steps", 5, 100, 25)

network_type = st.sidebar.selectbox(
    "Network Type",
    ["Erdos-Renyi", "Barabasi-Albert", "Watts-Strogatz"]
)

run_sim = st.sidebar.button("▶ Run Simulation")

# -------------------------------
# FUNCTIONS
# -------------------------------
def create_network(network_type, num_nodes, edge_prob):
    if network_type == "Erdos-Renyi":
        G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    elif network_type == "Barabasi-Albert":
        m = max(1, int(edge_prob * num_nodes / 2))
        G = nx.barabasi_albert_graph(num_nodes, m)
    elif network_type == "Watts-Strogatz":
        k = max(2, int(edge_prob * num_nodes))
        if k % 2 != 0:
            k += 1
        G = nx.watts_strogatz_graph(num_nodes, k, 0.2)
    else:
        G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Ensure graph is connected enough
    if len(G.edges()) == 0:
        G = nx.erdos_renyi_graph(num_nodes, 0.1)
    return G


def initialize_states(G, initial_believers, initial_factcheckers):
    states = {node: "S" for node in G.nodes()}

    all_nodes = list(G.nodes())
    believers = random.sample(all_nodes, min(initial_believers, len(all_nodes)))

    remaining = [n for n in all_nodes if n not in believers]
    factcheckers = random.sample(remaining, min(initial_factcheckers, len(remaining)))

    for b in believers:
        states[b] = "I"
    for r in factcheckers:
        states[r] = "R"

    return states


def simulate_step(G, states, degree_centrality, infection_prob, recovery_prob,
                  skeptic_to_fact_prob, threshold_fraction, centrality_boost):
    new_states = states.copy()

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            continue

        believer_neighbors = sum(1 for n in neighbors if states[n] == "I")
        fact_neighbors = sum(1 for n in neighbors if states[n] == "R")

        believer_fraction = believer_neighbors / len(neighbors)

        centrality_factor = 1 + (degree_centrality[node] * centrality_boost)

        # SKEPTIC -> BELIEVER or FACT-CHECKER
        if states[node] == "S":
            # Threshold-based misinformation adoption
            if believer_fraction >= threshold_fraction:
                adopt_prob = min(1.0, infection_prob * centrality_factor)
                if random.random() < adopt_prob:
                    new_states[node] = "I"

            # Skeptic may become fact-checker if many R neighbors
            if fact_neighbors > believer_neighbors and random.random() < skeptic_to_fact_prob:
                new_states[node] = "R"

        # BELIEVER -> FACT-CHECKER
        elif states[node] == "I":
            if fact_neighbors > 0:
                convert_prob = min(1.0, recovery_prob + 0.05 * fact_neighbors)
                if random.random() < convert_prob:
                    new_states[node] = "R"

        # FACT-CHECKER stays R
        elif states[node] == "R":
            pass

    return new_states


def count_states(states):
    counts = {"S": 0, "I": 0, "R": 0}
    for state in states.values():
        counts[state] += 1
    return counts


def draw_graph(G, states, degree_centrality):
    color_map = []
    size_map = []

    for node in G.nodes():
        if states[node] == "S":
            color_map.append("skyblue")
        elif states[node] == "I":
            color_map.append("red")
        else:
            color_map.append("green")

        size_map.append(300 + degree_centrality[node] * 3000)

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title("Social Network State")
    ax.axis("off")
    return fig


# -------------------------------
# MAIN SIMULATION
# -------------------------------
if run_sim:
    G = create_network(network_type, num_nodes, edge_prob)
    states = initialize_states(G, initial_believers, initial_factcheckers)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    history = []

    for step in range(steps + 1):
        counts = count_states(states)
        counts["Step"] = step
        history.append(counts)

        if step < steps:
            states = simulate_step(
                G, states, degree_centrality,
                infection_prob, recovery_prob,
                skeptic_to_fact_prob, threshold_fraction,
                centrality_boost
            )

    df_history = pd.DataFrame(history)

    # -------------------------------
    # LAYOUT
    # -------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🌐 Final Network Visualization")
        fig = draw_graph(G, states, degree_centrality)
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Final Counts")
        final_counts = df_history.iloc[-1]
        st.metric("Skeptics (S)", int(final_counts["S"]))
        st.metric("Believers (I)", int(final_counts["I"]))
        st.metric("Fact-checkers (R)", int(final_counts["R"]))

    # -------------------------------
    # LINE CHART
    # -------------------------------
    st.subheader("📈 Spread Over Time")
    st.line_chart(df_history.set_index("Step")[["S", "I", "R"]])

    # -------------------------------
    # CENTRALITY TABLE
    # -------------------------------
    st.subheader("⭐ Network Centrality Analysis")

    centrality_df = pd.DataFrame({
        "Node": list(G.nodes()),
        "Degree Centrality": [degree_centrality[n] for n in G.nodes()],
        "Betweenness Centrality": [betweenness_centrality[n] for n in G.nodes()],
        "Closeness Centrality": [closeness_centrality[n] for n in G.nodes()],
        "Final State": [states[n] for n in G.nodes()]
    })

    st.dataframe(centrality_df.sort_values(by="Degree Centrality", ascending=False), use_container_width=True)

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    st.subheader("🧠 Insights")
    max_believers = df_history["I"].max()
    peak_step = df_history["I"].idxmax()

    st.write(f"🔴 **Peak misinformation spread** reached **{max_believers} believers** at **step {peak_step}**.")
    st.write("⭐ Nodes with **high centrality** influence the spread more strongly.")
    st.write("⚠ If believer neighbors cross the **threshold fraction**, skeptics are more likely to adopt misinformation.")
    st.write("✅ Fact-checkers help reduce the number of believers over time.")

    # -------------------------------
    # DOWNLOAD CSV
    # -------------------------------
    st.subheader("⬇ Download Results")
    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Simulation Data as CSV",
        data=csv,
        file_name="misinformation_simulation.csv",
        mime="text/csv"
    )

else:
    st.info("Set parameters from the sidebar and click **Run Simulation**.")