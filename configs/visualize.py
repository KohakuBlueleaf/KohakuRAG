"""Config for visualize_embeddings.py"""

from kohakuengine import Config

db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
output = "artifacts/embedding_viz.png"

# Dimensionality reduction method: "pca", "tsne", "umap"
method = "umap"

# Node types to display (None = all types)
# Options: "document", "section", "paragraph", "paragraph_full", "paragraph_avg",
#          "sentence", "image"
show_types = None  # Show all

# Flat sampling: Maximum samples per node type (None = all)
max_samples = None

# Hierarchical sampling: sample based on tree structure
# Set hierarchical_sampling = True to enable
hierarchical_sampling = True
samples_per_document = 8  # Sections per document (None = all)
samples_per_section = 8  # Paragraphs per section (None = all)
samples_per_paragraph = 8  # Sentences per paragraph (None = all)

# t-SNE parameters
perplexity = 30

# UMAP parameters
n_neighbors = 32
min_dist = 0.1
n_jobs = -1  # Parallel jobs for UMAP (-1 for all cores)

# Random seed
random_state = 42

# Figure size
figsize = (12, 8)

# Color mode: "type" (color by node type) or "document" (color by document)
# When "document", shape indicates node type
color_mode = "document"

# Axis limit percentile: clip outliers by using percentile-based axis limits
# e.g., 99 means use 1st-99th percentile range (clips top/bottom 1%)
# Set to None or 100 to disable (use full range)
axis_percentile = 99


def config_gen():
    return Config.from_globals()
