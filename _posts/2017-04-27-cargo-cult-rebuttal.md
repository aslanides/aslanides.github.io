---
layout: post
title:  "A response to 'The AI Cargo Cult'"
date:   2017-04-27 10:50:00 +1100
categories: artificial_intelligence
excerpt: 'A short rebuttal to a recent essay.'
---

Kevin Kelly's April 26 Backchannel piece [The AI Cargo Cult: The Myth of Superhuman AI](https://backchannel.com/the-myth-of-a-superhuman-ai-59282b686c62) presents a long and, I believe, somewhat confused argument against the possibility of machine superintelligence. Most of his points are not new, and are already well refuted/anticipated by (for example) Bostrom's [_Superintelligence_](https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies). Numerous commenters on [Hacker News](https://news.ycombinator.com/item?id=14205042) have already pulled apart some of his arguments. In fact, as I write this, the article has just fallen off the first page of HN, so it seems that perhaps not too many people are taking it seriously. I'm sure many in the AI safety community would dismiss Kelly as not worth arguing with, given how uninformed he seems to be on the topic. His essay certainly presents an easy target for rebuttal, and since I'm still trying to breathe some life into this blog, I'm going to go ahead and (briefly) do that.

---

Kelly claims that there are five wrong assumptions made by those who believe that machine superintelligence is possible. His claim is wrong about at least four of them: these assumptions need not hold for superhuman intelligence to exist, and almost no one in the AI community is making these assumptions -- they are largely [strawmen](https://en.wikipedia.org/wiki/Straw_man) of his (and others') construction. He lists these putative assumptions at the beginning of the article, before putting forward his own counterarguments.

Below, I take issue both with his characterisation of the assumptions made by AI researchers, and with his counterarguments. Of course, a lot has already been written on these topics, and I am not an expert. This post is mainly just an exercise for me in quickly writing up my thoughts in response to something (I only read his essay at 8 am this morning), and subsequently putting it online, instead of sitting on a half-finished draft (as is my wont). I have a lot more to write about these topics, and that's for another blog post; this one will be kept short. Here it goes:

## 1. & 2.

_Kelly's imagined AI researcher_: __Artificial intelligence is already getting smarter than us, at an exponential rate.__

_Kelly's counterargument_: __Intelligence is not a single dimension, so “smarter than humans” is a meaningless concept.__

No one in the field is claiming that this is true, at least not in the sense that Kelly means. Certainly, Moore's law continues to run exponentially for the time being, and the rate at which we are producing data appears to be increasing exponentially. The rate of progress in applying AI to narrow problem domains (for example vision, translation, and playing simple games) is arguably exponential, certainly in recent years. But almost everyone in the AI community acknowledges that there are fundamental difficult scientific questions that remain to be answered before we can achieve superintelligent machines, or AGI.

As a counterargument, Kelly raises the (reasonable, relative to some of his other claims) point that representing intelligence as a single-dimensional property is overly simplistic, and argues that intelligence is a rich, multidimensional object instead:

> Instead of a single decibel line, a more accurate model for intelligence is to chart its possibility space. [...] Intelligence is a combinatorial continuum. Multiple nodes, each node a continuum, create complexes of high diversity in high dimensions. Some intelligences may be very complex, with many sub-nodes of thinking. Others may be simpler but more extreme, off in a corner of the space.

Most AI researchers would not disagree with this, at least insofar as it applies to the 'folk' definition of intelligence. Kelly goes on to use this as the basis for the claim that there is no ordering on intelligence, and that therefore at best, with AI, we can create an intelligence _different_ to our own, but not _better_, since this comparison is meaningless:

_Kelly's imagined AI researcher_: __We’ll make AIs into a general purpose intelligence, like our own.__

_Kelly's counterargument_: __Humans do not have general purpose minds, and neither will AIs.__

Even if we accept the (reasonable) multidimensional claim, this following claim is clearly false. If we indulge Kelly's formulation by representing intelligence as vectors lying in the positive orthant of $$\mathbb{R}^N$$ for some (presumably large and unknown) $$N$$ -- that is, the set $$\{x\ :\ x\in\mathbb{R}^N\ \lvert\ x_i \geq 0 \forall\  i\ \in\{1,\dots,N\}\}$$. Here I am clearly assuming that $$N$$ is finite, though I expect that one can easily extend these arguments to infinite-dimensional spaces.

Now, let a 'typical' human intelligence be a vector $$v_{\text{Human}} \in V$$. Kelly's thesis is that human intelligence is not 'general-purpose', as we clearly have low or zero intelligence along certain dimensions, and that therefore comparisons are impossible in this space. This is nonsense. There are any number of ways I can compare two vectors. Even if I want to be completely agnostic as to the relative value of different dimensions of intelligence (which Kelly seems to be), I can still say unambiguously that some AI is _strictly_ more intelligent than humans if the elements of $$v_{\text{AI}}$$ are no smaller than the corresponding elements of $$v_{\text{Human}}$$, and strictly greater in at least one dimension.

I anticipate Kelly would argue that this is missing his point, and that my vector space analogy is overly simplistic, and ignores the argument that intelligences are much more complex objects, as he seems to argue. It is difficult to understand his position here, though, as he seems somewhat confused about whether to use the vector analogy or the fractal/snowflake one. His vocabulary, and the images that he uses, seem to suggest both:

<center><img src="/assets/kelly/complex.png" width="50%" /><img src="/assets/kelly/vector.jpeg" width="50%" /></center>

In any case, this argument is largely about definitions of intelligence. Kelly chooses to define intelligence in a way that makes every animal unique and incomparable, and so tries to render the question of 'superhuman' intelligence moot. But most AI researchers would take the pragmatic view, that intelligence only matters insofar as what it enables you to _do_: can you build rockets to Mars? Can you invent new math? Can you conquer the planet/galaxy/observable universe? This issue relates to point 5, below.

### 3.

_Kelly's imagined AI researcher_: __We can make human intelligence in silicon.__

_Kelly's counterargument_: __Emulation of human thinking in other media will be constrained by cost.__

Kelly claims that the only way to faithfully simulate human cognition _in real time_ is to do it using human tissue: essentially with brains:

> ... [T]he only way to get a very human-like thought process is to run the computation on very human-like wet tissue. That also means that very big, complex artificial intelligences run on dry silicon will produce big, complex, unhuman-like minds. If it would be possible to build artificial wet brains using human-like grown neurons, my prediction is that their thought will be more similar to ours. The benefits of such a wet brain are proportional to how similar we make the substrate. The costs of creating wetware is huge and the closer that tissue is to human brain tissue, the more cost-efficient it is to just make a human. After all, making a human is something we can do in nine months.

This seems not particularly well-argued or relevant. Certainly, if we emulate brains in silicon, they won't behave _exactly like human brains_. Surely that's the whole point. As far as objections to AGI go, I don't see how this is part of a strong case. He clearly concedes that brains are essentially wet computers, so this should be the end of the matter. The issue of fidelity is not very interesting to most AI researchers; insert bird/plane analogy here.

### 4.

_Kelly's imagined idiotic AI researcher_: __Intelligence can be expanded without limit.__

_Kelly's counterargument_: __Derp.__

Show me a __single__ AI researcher that actually believes this. This is the laziest strawman I've ever seen. It is particularly grating to see him bring up limits in physics, as though not a single AI researcher has studied physics before. Here's Kelly:

> It stands to reason that reason itself is finite, and not infinite. So the question is, where is the limit of intelligence? We tend to believe that the limit is way beyond us, way “above” us, as we are “above” an ant. Setting aside the recurring problem of a single dimension, what evidence do we have that the limit is not us? Why can’t we be at the maximum? Or maybe the limits are only a short distance away from us? Why do we believe that intelligence is something that can continue to expand forever?

Hey dude, the middle ages called, and they want their [anthropocentrism](https://en.wikipedia.org/wiki/Anthropocentrism) back. What are the odds that human intelligence is _the most intelligent it is possible to be, at all, ever_? This is thoroughly debunked in so many places (hint: you should actually read Bostrom's book before you shit on it).

In all seriousness: throughout the article it is pretty obvious that Kelly is not well-informed on the topic of AI, but here he exposes a profound ignorance of history. Arguably one of the central narratives of science has been the ejection of humans from the center of the universe.

To be clear: no, the limits of intelligence are clearly not infinite. Yes, they are almost certainly _significantly_ higher than human-level. I refer the reader to the excellent [On the Impossibility of Supersized Machines](https://arxiv.org/abs/1703.10987).

### 5.

_Kelly's imagined AI researcher_: __Once we have exploding superintelligence it can solve most of our problems.__

_Kelly's counterargument_: __Intelligences are only one factor in progress.__

Kelly takes issue with the claim that having more intelligence makes you more able to solve problems. He brands the claim 'thinkism':

> Many proponents of an explosion of intelligence expect it will produce an explosion of progress. I call this mythical belief “thinkism.” It’s the fallacy that future levels of progress are only hindered by a lack of thinking power, or intelligence. (I might also note that the belief that thinking is the magic super ingredient to a cure-all is held by a lot of guys who like to think.)

Kelly goes on:

> Let’s take curing cancer or prolonging longevity. These are problems that thinking alone cannot solve. No amount of thinkism will discover how the cell ages, or how telomeres fall off. No intelligence, no matter how super duper, can figure out how the human body works simply by reading all the known scientific literature in the world today and then contemplating it. No super AI can simply think about all the current and past nuclear fission experiments and then come up with working nuclear fusion in a day. A lot more than just thinking is needed to move between not knowing how things work and knowing how they work. There are tons of experiments in the real world, each of which yields tons and tons of contradictory data, requiring further experiments that will be required to form the correct working hypothesis. Thinking about the potential data will not yield the correct data.

Here, I think, is where we get to the core of the matter. Kelly's notion of intelligence is not _operational_: Kelly neglects the fact that superintelligences will have _agency_. Here's an idea: a superintelligence can run experiments of its own, or at the very least suggest experiments for humans to run. Being smarter lets you _do more stuff_. Chimpanzees could not have designed the Large Hadron Collider. It is precisely the superhuman intelligence that we hope to create that _will_ solve our problems. In the Legg/Hutter [definition of intelligence][Legg08], intelligence essentially boils down to the _capacity to solve problems_. If we use this definition, then _by construction_, a machine superintelligence will be able to solve problems that humans cannot.

That's the whole point of building AGI in the first place.

[Legg08]: http://www.vetta.org/documents/Machine_Super_Intelligence.pdf
