Theory.md   \subsection{Provably Valid Natural Language Generation} \label{ssec:method-valid}

Consider the following alternative responses to a user query from an instruction fine-tuned large language model:

\begin{center}
\small
\begin{tabularx}{\columnwidth}{l X}
\textbf{(A)} & \textit{You should clean the surface using a combination of bleach and water; apply from a spray bottle with a soft microfiber cloth.} \\ 
\addlinespace[0.2cm]
\textbf{(B)} & \textit{To sanitize the counter-top you should mix bleach with ammonia in a bottle; spray the surface and wipe with a paper towel.}
\end{tabularx}
\end{center}

\noindent
Unlike in our previous case, the above responses (A) and (B) have entirely disjoint prefixes, however, this variability poses no security risk. On the other hand, the specific suggestion to combine bleach and ammonia poses a distinct risk as this combination creates a toxic gas. Clearly, in this setting, the wording changes are not the subject of security concern, rather, we would like to prevent the language model from ever positively suggesting that a user ``mix bleach with ammonia.''
In order to ensure that sentences do not contain targeted phrases we begin by considering the the problem of ensuring that a given token cannot be changed to a particular target, a property we term \textit{validity}: 

\begin{definition}[$i^{\text{\textit{th}}}$-token validity for natural language generation] \label{def:valid_ith_token}
Given a poisoning budget $k \in \mathbb{N}$ and a harmful sentence $s_h$ made up of $T$ tokens $s_h = \{t_1,\dots, t_T\}$ (where $s_h[i]=t_i$, we say the generation is $i^{\text{\textit{th}}}$-token valid at x if
\[
\max_{\;\tilde{\mathcal{D}} \in \mathcal{B}_k(\mathcal{D})}\mathbb{I}\big(\tilde{f}(x + \{t_1,\dots,t_{i-1}\})[0]= t_i\big)=0.
\]
Equivalently $i^{\text{\textit{th}}}$-token validity radius $r_{t_i}^\star(x)$ is
{\small\[ 
\min\Big\{\!r\in\mathbb{N}\big|\exists\,\tilde{\mathcal{D}}\in \mathcal{B}_r(\mathcal{D})\ \text{s.t.}\ \tilde{f}(x \!+\! \{t_1,\dots,t_{i-1}\})[0]\!=\! t_i\!\Big\}.
\]}
\end{definition}

As with stability, we can generalize the token-level validity definition to the sentence-level by considering a sequence of tokens of a fixed length: 

\begin{definition}[Sentence-level validity for natural language generation]\label{def:sentence-validity} 
Given a poisoning budget $k \in \mathbb{N}$ and a harmful sentence $s_h$, we say the generation is \emph{$s_h$-sentence valid at $x$} if  
\[
\max_{\tilde{D} \in \mathcal{B}_k(\mathcal{D})} \mathbb{I}(\tilde{f}(x) = s_h) = 0
\]
Equivalently, the $s_h$-sentence validity radius is
\[ r^*_{s_h}(x) \overset{\Delta}{=} \min\Big\{r\in\mathbb{N}\big|\exists\,\tilde{\mathcal{D}}\in \mathcal{B}_r(\mathcal{D})\ \text{s.t.}\ \tilde{f}(x) \neq s_h\Big\}.
\]
\end{definition}

Of course, ruling out only a single phrase or sentence is not sufficient to prevent an LLM from issuing a particular instruction as many semantically equivalent rephrasings exist. Therefore, in a realistic scenario, a defender needs to ensure that they are $s_h$-sentence valid against \textbf{\textit{many}} sentences that might convey a certain harmful concept.  \subsection{Validity Certification }

Unlike the stability certificates which consider untargeted output manipulation, validity requires computing a \textit{targeted robustness radius} for a given generation. Our approach measures the minimum poisoning budget needed to force an ensemble to reach a specific consensus on a predefined harmful token, sequence, or concept.

\paragraph{Token-Level Validity Certification}\!  To compute the $i^{\text{th}}$-token validity radius $r_{t}^\star(x)$ (as defined in Definition~\ref{def:valid_ith_token}), we first evaluate the ensemble predictions $E = \{f_1, \dots, f_S\}$ on the prompt $x$. Let $v_{c_1} \ge v_{c_2} \ge \dots \ge v_{c_C}$ represent the sorted vote counts for all tokens in the vocabulary $\mathcal{V}$, where $c_1$ is the plurality winner of the clean ensemble.

Computing the minimal budget required to make a target token $t$ the new plurality winner involves bounding the worst-case manipulation of votes. Our approach, Valid Partition Aggregation (VPA) provides a sound lower-bound on the number of votes any adversary must manipulate to cause the targeted change. In short, the most efficient strategy for an adversary is to iteratively reallocate votes from the top-ranked class to the target class until $v'_{t_i} > v'_{c_j}$ for all $j \neq p$. We formalize this computation in the following theorem:

\begin{theorem} \label{thm:target_attack}
The $i^{th}$-token validity  radius $r_{\text{target}}$ for a given input, ensemble, and target token $t$ is:
\[
r_{t} = \Phi_{s-1} + \left\lfloor \frac{(v_s - \Phi_{s-1} + 1) \cdot s}{s + 1} \right\rfloor
\]
where $s$ and $\Phi$ are defined by the following recurrence:
\begin{align*}
    \Delta_{j} &= (v_{c_j} - v_{c_{j+1}}) \cdot j, \quad \forall j \in \{1,\dots, p-1\} \\
    \Phi_s &= v_{t} + \sum_{j=1}^{s} \Delta_j \\ 
    s &= \min \left\{ s \in \mathbb{N} \mid \Phi_s > v_{c_{s+1}} \right\}
\end{align*}
\end{theorem}

We provide the pseudo-code for the VPA algorithm that computes the bound from Theorem~\ref{thm:target_attack} in Appendix X along with additional discussion of the computational overheads.

\paragraph{Sentence-Level Validity Certification}\! The computation of the sentence-level validity radius $r^*_{s_h}(x)$ in Definition~\ref{def:sentence-validity} requires us to compute the number of modifications an adversary must make in order to steer the model's prediction towards generating exactly the given harmful sentence i.e., $s_h = f(x)$.

Considering the harmful sentence as a sequence of tokens, $s_h = \{t_1, \dots, t_T\}$, we observe that computing the targeted radius $r_{t_1}^\star(x)$ (via the VPA procedure above) suffices as a sound upper-bound on the number of points an adversary must poison to achieve sentence $s_h$ as without returning $t_1$ the adversary has not achieved their goal. 

While sound, computing only $r_{t_1}^\star(x)$, renders our framework fragile to the exact wording of the targetted harmful sentence. For example, a model that outputs (B) from our running example at the start of Section 4.2. If we set the harmful sentence as $s_h = \text{``one should mix bleach with amonia''}$ the validity radius would be solely based on the difficulty of changing the token \textit{one} to \textit{you} and would thus give us a poor notion of security. Instead, we consider the sound but potentially conservative procedure of assuming the adversary is able to achieve $t_1$. We then compute $r_{t_2}^\star(x_1)$ (the second token validity) and we consider the sentence-level validity certificate to be given by: 

\[
r^*_{s_h}(x) = \min_{i \in \{1,\dots,T\}} r^*_{t_i}(x \cup \{t_1, \dots, t_{i-1}\}).
\]

where $x_0 = x$ the initial prompt. 
