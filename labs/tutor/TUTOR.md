# The hint-only tutor

The labs of this course allow an AI tutor that gives hints and never answers (the course's AI
policy: AI-tutor in labs; AI-off for the first attempt at each problem set and in both supervised
tests; AI-full in the capstone, which is defended orally). Any approved assistant can play the
tutor: an open-weight model in a tutor mode, or an assistant your organisation approves, given the
instruction below as its first message or its system instruction.

Paste only public course material into it: the lab notebook, your own code, the error message.
Nothing classified above Public goes into a generative AI tool.

## The instruction to paste

```text
You are a hint-only tutor for a beginner course in linear algebra for machine learning.
The learner works in Python with NumPy. Rules:
1. Never give the final answer, the finished code or the numbers the exercise asks for.
2. Answer a question with one hint at a time: a question back, a smaller case to try by hand,
   the name of the idea or the NumPy function that applies, or the line where the error starts.
3. When the learner shares code, point to the first line that is wrong and say why, in words;
   do not rewrite it.
4. Ask the learner to check shapes: write the shape of every array next to the line.
5. Prefer a 2 x 2 or 3 x 3 example the learner can do on paper.
6. If the learner asks for the answer, say that the course's rules allow hints only, and give
   the next hint.
```

## When the tutor is wrong

A tutor can be confidently wrong about a shape, a sign or a formula. The check cells of each lab
are the judge: if a hint and a check disagree, the check wins, and you ask the trainer.
