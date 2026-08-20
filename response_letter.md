# Response to the Associate Editor and Reviewers

We thank the Associate Editor and the two reviewers for their careful reading of our manuscript and their constructive comments. We have substantially revised the manuscript in response to every comment. The most significant changes are:

1. We added two recent, state-of-the-art benchmark algorithms across all six experimental settings: **MOPSO-AMS**, an adaptive multi-swarm evolutionary heuristic that embeds cooperative strategies and heuristic information (Han et al., 2023), and **CAPQL**, a preference-conditioned multi-objective reinforcement learning (MORL) algorithm (Lu et al., 2023). All twelve result tables, the accompanying figures, and the Discussion section have been regenerated and rewritten around this six-algorithm comparison.
2. We added a new subsection, "Conflicting Nature of the Objective Functions" (Section 3.2), that formally analyzes, via the partial derivatives of the three objectives with respect to the decision variables $(r,Q)$, whether and how the objectives conflict.
3. We added hypervolume (HV) and inverted generational distance (IGD) as evaluation metrics throughout, and restructured the Experiment section so that overall algorithm-superiority claims are based primarily on HV and IGD, with set coverage, spacing, and maximum spread retained as complementary diagnostics that explain the mechanisms behind the HV/IGD results.
4. We clarified the methodological connection between the two proposed PCN extensions (Section 4.3), addressing the concern that the framework could be read as a simple sequential combination of two existing methods.
5. We corrected the identified inconsistencies in Algorithm 2's pseudocode and clarified the definition and purpose of the mean objective values reported in the metric tables.

Reviewer comments are reproduced in italics below, followed by our point-by-point response.

---

## Response to the Associate Editor

> *1. When evaluating the proposed method against existing methods/approaches, more recent and state-of-the-art methods should be included for comparison (see Comments 1 and 3 by Reviewer 1 and Comment 3 by Reviewer 2).*

We agree this was the most significant gap in the original submission. We have added MOPSO-AMS and CAPQL as two additional, genuinely state-of-the-art benchmarks and re-run the full experimental protocol across all six demand settings. Our detailed response, including how these algorithms were selected and how the results change the paper's conclusions, is given under Reviewer 1, Comments 1 and 3, and Reviewer 2, Comment 3 below.

> *2. The authors need to provide a more detailed explanation of the three objective functions introduced on page 8, particularly regarding the functions' behaviour and whether/how they constitute a "conflicting" multi-objective optimization problem.*

We have added a new subsection (Section 3.2, "Conflicting Nature of the Objective Functions") that derives the partial derivatives of all three objectives with respect to both decision variables and uses them to establish precisely where and why the objectives conflict. Our detailed response is given under Reviewer 1, Comment 2 below.

We have also addressed all remaining comments from both reviewers, including Reviewer 1's Comment 4 and Reviewer 2's Comments 1 and 2, as detailed below.

---

## Response to Reviewer #1

> *1. For multi-objective optimization problems, numerous effective solution approaches have been proposed within evolutionary algorithms, including heuristic information utilization, reinforcement learning techniques, and others. In this paper, the authors adopt the PCN framework yet only elaborate on the deficiencies of traditional evolutionary algorithm frameworks, without conducting comparative analyses with state-of-the-art adaptive evolutionary algorithms embedded with cooperative strategies and heuristic information proposed in recent years. It is suggested that the authors further clarify the superiority of adopting the PCN framework through systematic analysis of the aforementioned advanced methods.*

We thank the reviewer for this suggestion. We have added **MOPSO-AMS**, an adaptive multi-swarm PSO variant that coordinates multiple cooperating sub-swarms and incorporates heuristic information to guide particle updates, as a benchmark across every experimental setting (Section 2, Literature review, now introduces and cites this algorithm; Section 5, Tables 1–12, report its full results). MOPSO-AMS is a substantially stronger evolutionary baseline than plain MOPSO: under stationary demand it achieves HV and IGD values close to our proposed Guided PCN and dominates a larger share of Guided PCN's solutions than the reverse in terms of set coverage. However, our expanded results show that this advantage does not generalize: under nonstationary demand it offers little improvement over plain MOPSO, and in the real-world case study it underperforms even plain MOPSO because its evolutionary search over-prioritizes cost at the expense of service level. This directly demonstrates, with a genuinely state-of-the-art cooperative/heuristic-informed evolutionary algorithm as the comparison point, that PCN's adaptability advantage is not an artifact of comparing against a weak evolutionary baseline. This analysis is presented in the revised Discussion section (Section 5.5).

> *2. The three objective functions proposed in Eqs(1-4) require further elaboration. In particular, it is necessary to verify and illustrate whether these objectives are "conflicting" in multi-objective optimization.*

We have added a new subsection, Section 3.2, "Conflicting Nature of the Objective Functions," directly after the objective functions are introduced. In it, we derive the partial derivatives of total cost $C(r,Q)$, stockout frequency $N(r,Q)$, and stockout amount $F(r,Q)$ with respect to both decision variables, independent of the specific lead-time demand distribution $f_{D_L}$. We show that:
- With respect to $r$: $C$ increases monotonically while $N$ and $F$ decrease monotonically, so the three objectives are unconditionally conflicting with respect to $r$.
- With respect to $Q$: $N$ and $F$ still decrease monotonically, but $C$ is convex with a unique minimum at the classical EOQ point $Q^\ast = \sqrt{2AD/h_c}$; conflict with respect to $Q$ therefore only emerges once $Q$ exceeds $Q^\ast$.
- We further clarify that $N$ and $F$ arise from the same underlying stockout event and tend to move together under a policy change, so they should not be interpreted as conflicting with each other; rather, they capture complementary facets (frequency versus severity) of service performance, while both remain in conflict with cost.

This provides a rigorous, closed-form justification for treating the problem as multi-objective, independent of any particular demand distribution.

> *3. The comparative analysis of the proposed algorithm is insufficient. The authors should supplement comparisons with the latest multi-objective evolutionary algorithms, such as multi-population adaptive cooperative evolutionary algorithms. Moreover, by benchmarking against the commonly-used evaluation metrics in multi-objective optimization, namely HV and IGD, the authors are required to demonstrate the advantages of the three evaluation metrics adopted in this study.*

We have added MOPSO-AMS (a multi-population, adaptive, cooperative evolutionary algorithm, as requested) and CAPQL (a state-of-the-art MORL algorithm) as benchmarks, and we now report HV and IGD alongside set coverage, spacing, and maximum spread in every result table (Section 5.1, "Performance metrics for Pareto front evaluation," now defines HV and IGD formally, in addition to the three original metrics). We restructured the interpretive text throughout Section 5 so that HV and IGD are the primary basis for judging overall front quality, since they are the most widely used convergence-and-diversity indicators in the multi-objective optimization literature.

At the same time, our results show that the three original metrics retain genuine diagnostic value beyond what HV/IGD alone reveal. For example, under stationary demand, MOPSO-AMS achieves a stronger set-coverage score than Guided PCN despite trailing it on HV and (mostly) IGD; examining spacing and maximum spread shows that MOPSO-AMS's front is more tightly clustered and narrower than Guided PCN's, explaining this apparent discrepancy as a difference in search behavior (local dominance versus broad, well-converged coverage) rather than a contradiction. We now make this complementary role explicit in the revised Section 5.2 and 5.3, using set coverage/spacing/maximum spread specifically to explain *why* certain HV/IGD patterns arise, rather than as independent superiority claims.

> *4. Further explanations should be provided for the three "mean" values presented in Tables 4-6, 10 and 12. Specifically, how can these mean values be obtained? Also, the significance of such comparative analysis in the results of multi-objective optimization need to be explicitly clarified.*

We have added the following clarification to Section 5.1: "In addition to these five front-level metrics, the mean value of each objective is also reported in the corresponding result tables to allow the trade-off relationships among the objectives to be examined from a practical standpoint. Specifically, for each metric and for the per-objective mean values, a value was first computed within each run from the non-dominated solutions obtained in that run, and the resulting ten values were then averaged across runs." We clarify the significance as follows: front-level metrics such as HV and IGD summarize the overall approximation quality of a Pareto front as a single number, but they do not by themselves reveal the realized cost and service levels a decision maker would actually experience. Reporting the per-objective means alongside the front-level metrics allows readers to connect abstract front-quality rankings to concrete, practically interpretable cost-service trade-offs.

---

## Response to Reviewer #2

> *1. The proposed Guided PCN is a two-stage pipeline. MOPSO runs first and produces a set of Pareto-optimal solutions. These solutions are then used as static initial samples for PCN. The interaction between the two components is not clearly described. The authors should clarify the core contribution of the framework at the theoretical or methodological level. They should explain how the two operators are connected and how they work together. Otherwise the framework may look like a simple sequential use of two existing methods.*

We agree the original manuscript did not make this connection explicit. We have added the following paragraph at the start of Section 4.3 ("Guided PCN"):

*"Although the two variations act at different steps of the PCN pipeline, they are unified by a common design principle rather than being independent, sequentially applied methods. The original PCN contains two distinct sources of undirected randomness: the initial buffer is built from random policies (Step 1), and the target-return perturbation used during exploration is likewise drawn without guidance (Step 3). The two proposed variations replace exactly these two sources with information-driven counterparts, while the network-training and buffer-maintenance steps (Steps 2 and 4) are left unchanged. Evolutionary buffer initialization determines where the search begins: it seeds the buffer with MOPSO-derived Pareto-optimal anchors, so that the desired-return targets PCN subsequently conditions on already correspond to high-quality, diverse trade-offs rather than to arbitrary random-policy outcomes. Guided exploration then determines how the search proceeds from these seeded targets: at every training cycle, it computes statistics directly from the current buffer -- which, in early training, is still dominated by the MOPSO-seeded solutions -- to decide which objective to improve and by how much. Because guided exploration is applied at every training cycle to a buffer whose composition and statistics are shaped by the evolutionary seeding throughout training, rather than being applied once as a preprocessing step before an otherwise unmodified PCN, the two mechanisms remain coupled for the entire training process rather than constituting a simple sequential combination of two existing methods."*

In short, MOPSO is not merely a one-time preprocessing step whose output is handed off to an otherwise-unmodified PCN: the statistics that drive guided exploration throughout training are computed from a buffer whose composition continues to reflect the MOPSO-derived seeding, so the two mechanisms jointly and continuously shape the search rather than acting as two disconnected stages.

> *2. The pseudocode of Algorithm 2 contains several inconsistencies. For example, "noise_scale" in Step 10 is never defined, "min_noise_scale" defined in Step 4 is never used. The authors should carefully revise the pseudocode.*

We have revised Algorithm 2. `noise_scale` is now explicitly computed as its own step, $noise\_scale = \max(\epsilon_j \times |a_{\text{base}}|,\ min\_noise\_scale)$, immediately before it is used to construct the initial action, and this computation explicitly uses `min_noise_scale` (defined earlier in the algorithm) as a floor to prevent the noise scale from collapsing to zero when $\epsilon_j$ is small. We also revised the subsequent per-timestep perturbation step for clarity and consistency with the surrounding notation.

> *3. The comparison baselines are outdated. Only MOPSO and the original PCN are compared, neither of which represents the current state of the art. The authors should include more recent multi-objective EAs and MORL algorithms to strengthen the experimental evaluation.*

As described in our response to Reviewer 1, Comments 1 and 3, we have added MOPSO-AMS (a recent, adaptive multi-swarm evolutionary heuristic) and CAPQL (a recent, preference-conditioned MORL algorithm) as benchmarks across all six demand settings. Both are described and cited in the revised Literature review (Section 2) and the Experiment section (Section 5). The revised Discussion (Section 5.5) explicitly summarizes where each of the two new state-of-the-art baselines is and is not competitive with the proposed Guided PCN, providing a substantially strengthened experimental evaluation.

> *4. The Pareto front evaluation relies on set coverage, spacing, and maximum spread. However, hypervolume (HV) and inverted generational distance (IGD) are widely adopted in multi-objective optimization and should be included to provide a more comprehensive evaluation.*

We have added both HV and IGD to Section 5.1 and to every result table in the revised Experiment section, and we now use them as the primary basis for comparing overall Pareto front quality across all six algorithms, as detailed in our response to Reviewer 1, Comment 3 above.

> *5. The authors should ensure notation and terminology consistency throughout the manuscript. For example, the reorder point is denoted r in Section 3.1 but k in Figure 6 and Section 5.3, and "tradeoff" and "trade-off" alternate throughout. The authors should carefully proofread the paper to unify all symbols and terminology.*

We thank the reviewer for catching this. We have standardized the reorder-point notation to $r$ throughout, including the Figure 6 caption and axis labels, which previously used $k$, and standardized "trade-off" (hyphenated) throughout the manuscript.

---

We believe these revisions directly and comprehensively address the concerns raised by the Associate Editor and both reviewers, and we thank them again for their valuable feedback, which has substantially strengthened the paper.
