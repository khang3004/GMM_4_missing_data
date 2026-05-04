# Gaussian Mixture Model Clustering with Incomplete Data

## Overview
- **Topic**: Gaussian Mixture Models (GMM) handling missing tabular data via joint optimization.
- **Hook**: "Missing values do not just remove numbers. They bend the cluster geometry."
- **Target Audience**: Data scientists / ML students who know basic probability (Gaussians, expected value) and clustering (KMeans/GMM).
- **Estimated Length**: 3-5 minutes
- **Key Insight**: Imputing data *before* clustering destroys separation. The mathematically elegant solution is to let each Gaussian cluster propose a local completion, weighted by its responsibility, updating both the model and the missing data simultaneously.

## Narrative Arc
We start with a visually intuitive problem: showing how naive imputation (like mean filling) collapses data points and ruins cluster geometry. Then we build up the solution by introducing the GMM's joint optimization view, where missing values aren't just filled; they are treated as variables that move towards their responsible clusters. Finally, we witness the beautiful alternating EM cycle dynamically reconstructing the data while finding the clusters.

---

## Scene 1: The Illusion of Imputation (The "Wrong" Approach)
**Duration**: ~45 seconds
**Purpose**: To establish the problem with two-stage pipelines (impute first, cluster later) and hook the viewer.

### Visual Elements
- 2D coordinate system (`Axes`).
- Two distinct clusters of data points (`Dot`s), colored differently (e.g., techBlue and alertRed).
- A subset of points in both clusters will have their x or y coordinates "lost" (represented by dropping to an axis or fading out one dimension).
- Animation sequence:
  1. Show full data $\to$ drop some coordinates.
  2. Apply "Mean Imputation" $\to$ the lost points snap to a straight line (the mean of the axis).
  3. Show how the two clusters are now horribly mixed along that line.

### Content
Show a scatter plot of two clean clusters. Suddenly, random coordinates go missing. The points "fall" onto the x or y axis, indicating we only know one coordinate. If we naively impute by filling with the mean, those points snap to a straight line. Visually, the clear separation between the red and blue clusters is completely destroyed.

### Narration Notes
"What happens when data goes missing? Your first instinct might be to just fill in the blanks—maybe use the average. But watch what happens to our distinct groups. By guessing blindly before we understand the groups, we destroy the very structure we're trying to find."

### Technical Notes
- Use `Transform` to move points to axes.
- Use `color` to highlight the points with missing data (e.g., turn them gray when missing, then back to their cluster color later).

---

## Scene 2: The Two Hidden Layers (Formulation)
**Duration**: ~60 seconds
**Purpose**: Introduce the GMM joint optimization perspective.

### Visual Elements
- Math equations (`MathTex`) appearing using Progressive Disclosure.
- Split screen: Math on left, Geometry on right.
- Highlight the missing data variable $x_{j,m}$.

### Content
Instead of filling the blanks first, what if the clustering algorithm fills them as it learns? We introduce the GMM log-likelihood, but with a twist: the missing entries $x_{j,m}$ are variables we optimize over. The observed data is "locked" in place, but the missing coordinates are free to slide along their unknown axes. 

### Narration Notes
"The solution is to not separate imputation and clustering. Optimize them together. In a standard Gaussian Mixture Model, we have one hidden layer: which cluster does a point belong to? With missing data, we have a second hidden layer: where does this point actually sit along its missing dimension? We lock the observed values, but let the missing ones slide."

### Technical Notes
- Use `TransformMatchingTex` to build the likelihood equation.
- Use `set_color_by_tex` to color the observed data $x_{j,o}$ BLUE and missing data $x_{j,m}$ YELLOW.

---

## Scene 3: The Precision-Weighted Update (M-step data)
**Duration**: ~60 seconds
**Purpose**: Visualize the core mathematical insight—how missing values are updated.

### Visual Elements
- A single data point missing its y-coordinate (constrained to move on a vertical line).
- Two Gaussian distributions (represented by contour ellipses).
- Vectors/Arrows pointing from the Gaussian centers to the point's line.
- The equation for $x_{j,m}^{\star}$.

### Content
Focus on one missing point. It can only move up or down (y is missing). We have two Gaussian clusters. "Each Gaussian component proposes a local completion." Show a faint arrow from the red cluster pulling the point to its ideal spot, and a blue arrow from the blue cluster pulling it. The final position is a weighted average of these proposals, decided by the precision matrix and the cluster responsibility $\gamma_{jk}$. 

### Narration Notes
"Here is the star of the show. How do we update a missing value? Every Gaussian cluster looks at the point's known coordinates and says, 'If this point belongs to me, it should be right here.' The point is pulled by each cluster, weighted by how strongly the cluster claims it. It's not generic imputation; it's a cluster-aware geometric reconstruction."

### Technical Notes
- `TracedPath` or `Line` to show the vectors pulling the point.
- The equation for $x_{j,m}^{\star}$ should appear at the bottom (Golden Layout) and parts of it highlight as the arrows pull.

---

## Scene 4: The Alternating Cycle (Algorithm & Payoff)
**Duration**: ~60 seconds
**Purpose**: Show the full EM loop converging.

### Visual Elements
- Full dataset from Scene 1, with missing points currently at bad estimates.
- Gaussian ellipses starting large and uninformative.
- A cyclic flowchart in the corner: `E-step` $\to$ `M-step (params)` $\to$ `M-step (data)`.

### Content
We put it all together. The flowchart highlights `E-step` (points change color/opacity based on cluster assignment), `M-step params` (ellipses shift and resize), and `M-step data` (missing points slide along their constrained axes towards the clusters). Run this loop rapidly 5-10 times. The initially chaotic points elegantly slide into two perfectly separated clusters, recovering the original geometry.

### Narration Notes
"When we alternate these steps, magic happens. The clusters guess the points, the points update the clusters, and the missing values slide precisely into place. Every step mathematically guarantees an improvement, recovering the hidden geometry that naive imputation would have destroyed."

### Technical Notes
- Use a fast `run_time` and a `ValueTracker` to animate the EM loop iterations smoothly.
- The cyclical diagram can use `Indicate` to flash which step is currently running.

---

## Transitions & Flow

### Scene Connections
- **Scene 1 → Scene 2**: The chaotic "mean imputed" data fades to the background as the math takes center stage.
- **Scene 2 → Scene 3**: We zoom out of the full equation and zoom in on a *single* point from the background to demonstrate the update rule.
- **Scene 3 → Scene 4**: We zoom back out to the full dataset, now applying the rule to all points simultaneously.

### Recurring Visual Motifs
- **Axes Constraints**: Whenever a point has missing data, it visually lives on a constrained line (e.g., a vertical line if y is missing), showing its "freedom of movement."
- **Color Identity**: Red and Blue always represent the two Gaussian components.

---

## Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Primary 1 | techBlue | #0056A0 | Cluster 1, observed data elements |
| Primary 2 | alertRed | #D32F2F | Cluster 2, missing data elements |
| Background| Dark grey | #1C1C1C | Standard 3b1b background |
| Accent | Yellow | #FFFF00 | Highlighting equations / key movements |
| Neutral | White | #FFFFFF | Axes, text, standard equations |

---

## Mathematical Content

### Equations to Render
1. Joint Objective: $\max_{X, \theta} \sum_{j} \log \left( \sum_{k} \alpha_k \mathcal{N}(x_j | \mu_k, \Sigma_k) \right)$
2. Missing Data Update: $x_{j,m}^{\star} = (\sum_k P_{jk} \Lambda_{k,mm})^{-1} \sum_k P_{jk} (\dots)$
3. E-step: $\gamma_{jk} \propto \alpha_k \mathcal{N}(x_j | \mu_k, \Sigma_k)$

### Geometric Objects
- Bivariate Gaussian level curves (ellipses).
- Data points (`Dot`) and constraint lines (`DashedLine`).

---

## Implementation Order

1. **Scene 1** - Establishes the coordinate system and data structure.
2. **Scene 3** - Focuses on the math and movement of a single point (easier to implement than the full loop).
3. **Scene 4** - Integrates Scene 1 and Scene 3 logic into a full loop.
4. **Scene 2** - Heavy on MathTex formatting, can be done last to tie it all together.

### Shared Components
- `GaussianContour` class: Custom mobject to draw ellipses from a mean and covariance matrix.
- `ConstrainedDot` class: A dot that knows which axis it is allowed to move along.
