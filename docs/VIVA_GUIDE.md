# Explain the implementation independently

Use these prompts with the actual files and results open. Team explanations should reflect understanding and observed evidence, rather than memorized AI-written answers.

Current executed status: five baseline models, 15 TPE trials, tuned HGB, student-test HMM comparison, and all ten final evaluator HTTP requests are complete. Open `notebooks/PathSnap_final_demo.ipynb` for actual outputs. The frozen model was created before evaluator import; evaluator outcomes did not drive a model change. Be explicit that pointwise results are alignment proxies: final HMM mean-trip Top-1 is 86.89%, while pooled Top-1 is 75.96%. Trip 00000015 is weaker at 48.03%. The physical second-laptop rehearsal and its receiving-terminal screenshots remain team actions.

1. Trace row 0 of a `.track` file: why longitude comes first, what the timestamp means, and how its original row index survives candidate generation and decoding.
2. Pick an `.arcs` row. Locate both zero-based endpoint rows in `.nodes`. Show how that row becomes a directed metric LineString and why treating degrees as metres would fail.
3. Explain why a long road crossing a GPS point must be found even when its midpoint/endpoints are far away. Demonstrate the exact segment-distance spatial query.
4. Describe the 12 features. Which require the previous GPS observation? What happens at stops, the first row, or a large gap? Show that no reference, trip ID or arc ID is an input feature.
5. Explain why `.route` cannot simply be zipped with `.track`. Work through the monotone alignment recurrence `D[t,j] = distance(t,j)^2 + min(D[t-1,h], h<=j)`. Repeated arc IDs are separate route occurrences. What biases does this geometric weak supervision introduce?
6. Open a CV fold. Show its train/validation trip lists are disjoint. Explain why a random candidate-row split would leak neighboring points and candidates of the same observation. Explain the remaining risk of the same roads appearing in different trips.
7. Explain each model's objective: logistic negative log likelihood; Naive Bayes class-conditional Gaussian likelihood; tree Gini splits; forest bagged Gini trees; HGB binary log loss. Discuss a real validation/test difference for each using the report. Do not claim under/overfitting from textbook properties alone.
8. Explain candidate binary Macro F1 versus observation Top-1. Which denominator includes absent reference candidates? Which observations lack a usable aligned reference? Why can a high candidate accuracy hide failure on positive candidates?
9. Show one TPE trial after the random startup phase, its parameters and held-out-trip objective. Why does selecting the best CV trial not make its score an unbiased estimate of future performance?
10. Hand-compute Viterbi for two observations with two candidates: emission log scores, directed-network transition scores, maximum accumulated score and backpointer. Distinguish row-normalized transitions from raw distance penalties.
11. Show a no-candidate, disconnected-transition or GPS-gap reset. Why is an explicit route break more honest than drawing a straight line? How are intermediate arcs reconstructed when a valid directed shortest path exists?
12. Explain why strict adjacency may be low for valid sparse observations crossing several roads. Compare it with arc-set overlap, sampled-arc edit similarity and the reconstructed route parts. Neither diagnostic alone proves a correct physical route.
13. Trace a frontend Match click through CORS, `/match`, reference-blind prediction, server-only scoring and a reference-free Folium response. Find its request ID in the receiving-laptop terminal.
14. Identify what has actually run, which artifacts prove it, and what remains untested. Explain assistance used, how the team reviewed it, and what changes/experiments the team independently made.
