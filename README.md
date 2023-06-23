# SDP_partialinfo
SDP algorithm to obtain the minimum QFI or maximum metrological usefulness from given partial information as discussed in this article https://arxiv.org/abs/2306.12711

Kindly refer to the article for a detailed analysis of the methods discussed and their advantages and disadvantages.

1. We compare the different SDP methods in the Appendix A.3 of the paper and C.1 with Tóth's SDP method and the advantages and disadvantages of each of them in [this](SDP_comparison.py) code file.

2. We check the effect of a finite $\delta\theta$ in our proposed SDP approach which can be seen [here](Finite_dtheta.py).

3. Using our SDP method in different examples such as [One-axis twisting dynamics](Oneaxistwisting.py), [Dicke-states](Dicke_states.py), and [spin chain systems](Spinchain.py).

4. Finally, we have the [code](Plots_data.py) to plot the figures for the one-axis twisting dynamics case with the data present in the [Data](Data) folder labelled as [second moments](Data/linearSS_10,30) and [fourth moments](Data/nonlinearSS_10,30) for system sizes from $N = 10$ to $N = 30$.

5. We have rest of the data for the Dicke states analysis, spin-chain system, and comparison with the two proposed SDP methods in the [folder](Data).

6. The plots used in the article can be found in the [folder](Plots).

In case of any queries, please write to us at [send email](guillem.muller@icfo.eu).
