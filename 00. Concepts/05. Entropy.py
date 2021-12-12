"""
-----------------------------------------------------------------------------------------------------------------------
Entropy:
-----------------------------------------------------------------------------------------------------------------------
The entropy of a random variable is the average level of “information“, “surprise”, or “uncertainty” inherent
in the variable’s possible outcomes. It is the measure of uncertainty. As we increase uncertainty, Entropy reduces.

Probability p(x) and Surprise/Information h(x) of an event are inversely proportional.
h(x) = log2(1 / p(x)) or -log(p(x))
If p(x) = 1, h(x) = 0
If p(x) = 0, h(x) = undefined

Entropy is the Expected Value of Surprise.
Entropy = E p(x) * log(1/p(x)) -> Summation [prob of surprise * surprise]
        = -E p(x) * log(p(x))
If p(x) = 0.5, Entropy = 1 (highest entropy when no uncertainty or surprise)
-----------------------------------------------------------------------------------------------------------------------
"""
