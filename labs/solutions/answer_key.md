# Labs: what each one should show

The numbers below come from running the solution notebooks in full (not quick) mode with the committed data. Timings
depend on the machine; every other number is fixed by the data and the seed.

| Lab | Planted in the data | What the participant should find |
|---|---|---|
| 1.1 | Article vectors share a large common direction | Raw cosine search ranks the judged-best article first for 2 of 10 questions (top 3 for 3); centring and content words give 5 and 6. Mean pair cosine falls from 0.853 to about 0. Occupation words lean towards one gendered set or the other (for example nurse +0.130, engineer −0.066): record what that means for a search built on them |
| 2.1 | — | The vectorised product runs hundreds of times faster than the loop at n = 128; today's daily scoring is about 0.30 GFLOP against 192 GFLOP planned, 640 times more |
| 3.1 | `total_actions` is the exact sum of three other columns | Rank 4 of 5; the smallest singular value is 0; cond(XᵀX) is about 10¹⁷ with five columns and 3 × 10⁵ with four; solving the five-column system fails |
| 4.1 | Video minutes are almost 7.5 × the number of videos in the learner features | Classical Gram-Schmidt loses orthogonality first, modified next, Householder stays near machine precision |
| 5.1 | Video minutes and video count nearly collinear in the study-hours data | QR and the normal equations agree to about 10⁻¹² in 64-bit floats; the gap grows with collinearity; in 32-bit floats the normal equations move the second coefficient from 0.4228 to 0.4308 |
| 6.1 | A dangling page (P35), a link farm (P32 to P34 pointing at P31), an island (P36 to P40) | Without damping the island holds 38% of the rank; with damping 0.85 the iteration converges in 110 steps; raising the damping lifts the farm's share and P31's position |
| 7.1 | — | Rank 50 keeps the brick image within 3.1% relative error at 19.6% of the storage; two components hold 14.9% and 13.6% of the digits' variance; 13 components reach 80% |
| 8.1 | Ratings generated at true rank 3 plus noise, with a held-out set | Held-out error is lowest at rank 3 (0.5432 against a baseline of 0.6022); a rank-8 adapter on a 768 × 768 matrix trains 2.08% of the parameters; one-pass variance in 32-bit floats gives 0.0 where two-pass gives 2.5 |

The worked answers to the problem sets, the tests and the oral check's items are not in this repository.
