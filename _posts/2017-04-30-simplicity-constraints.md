---
layout: post
title:  "Simplicity is Complicated; Contraints bring Freedom"
date:   2017-04-30 13:37:00 +1100
categories: general
excerpt: 'Ruminations on Pike, Strunk, and White.'
---

I've recently been getting back into programming with [Go], which I haven't used since early 2016, when I was helping out on the backend at [Karma Wiki]. Over the past couple of days, I've been [porting](https://github.com/aslanides/aixigo) parts of [AIXIjs] into Go, with a view to making it a highly performant and scalable reference implementation by drawing on [Go's excellent concurrency](http://divan.github.io/posts/go_concurrency_visualize/) features. In a future post, I'll explain what I've learned about performance tuning and parallelism in Go.

Needless to say, I'm very impressed with the design of the language and the toolset surrounding it, and intend to build more projects with it. As part of my Go revival, I've watched a few talks by [Rob Pike], one of the co-creators of the langauge. These of course include the modern classic ["Concurrency is not Parallelism"](https://www.youtube.com/watch?v=cN_DpYBzKso). However, one talk in particular resonated with me: the presentation at dotGo 2015, titled ["Simplicity is Complicated"](https://www.youtube.com/watch?v=rFejpH_tAHM).

In "Simplicitly is Complicated", among other things, Rob talks at length about the decisions that went into designing Go's notoriously [small feature set](https://golang.org/doc/faq). He points out that in many other languages (e.g. C++, Java, etc.), there are many ways to do things. In Go, this is largely not the case: the language -- from code formatting to its idiosynchratic approach to OO to dependency management -- is very _constrained_. There is usually only one way to do something, and it is usually constrained to be quite simple, robust and performant. Between the compiler's static analysis and the toolchain (`go vet`, `go lint` and `go fmt`), you don't get much choice in the matter. This is a good thing.

For some programmers, of course, these constraints make using the language a pain in the arse [citation needed]. Rob's point is that in other, heavily feature-laden languages, programmer _productivity_ is significantly impacted by wasting time thinking about how to express an idea. "Should I use feature X or feature Y? What are the trade-offs?", and so on. In Go, by introducing constraints on how things are expressed, Pike argues that the programmer is freed to spend their thinking about what matters: the design and architecture of the software they are writing.

This is of course not a new notion in programming, and is not even a new notion in languages in general. Computer languages are tools for expressing certain kinds of ideas. _Human_ languages are themselves tools for expressing ideas. Of course, these ideas are somewhat less formal than those typically expressed in computer languages, and their purpose and usage is quite different, but the language analogy (naturally) holds. Arguably,  __English is the C++ of human languages__: it is hugely flexible and expressive, comprises an enormous vocabulary, is composed out of numerous other languages, is widely (ab)used, has many traps and gotchas, and is generally quite difficult to master.

Most _native_ English speakers struggle to wield the language well. It is very common for students (even at the undergraduate or -- occasionally -- graduate level) to have considerable difficulty expressing their ideas concisely and coherently. The language offers so much flexibility that it's common to see people agonize over the best words to use, rather than spending their thinking about how to distill their thoughts and ideas into language.

Here I hope that I have made the analogy with programming clear; these are both _creative_ endeavours in which we use the tools (i.e. languages) at hand to realize our ideas. From my personal experience, I have struggled with this issue in both domains. Historically, I have had enormous difficulty putting my thoughts down on paper in such a way that I was pleased with the results. Until around 2012, essay and report writing largely represented exercises in frustrated word-shuffling and hair-pulling to me. Even in my recent [AIXIjs] project, I estimate that probably about 30% of my [commits] were spent either in changing superficial features of the program, or in vacillating between various (clumsy) design choices.

I say _until 2012_, because that was the year in which I did [Honours] in physics, working under [C. M. Savage]; the Honours program was immensely challenging and stimulating, and for me, it was a year of considerable intellectual maturation. As the thesis deadline approached, I grew apprehensive of the task of writing a [100+ page document](http://aslanides.io/docs/honours_thesis.pdf). The inimitable [J. D. Close] recommended a booklet (almost a _pamphlet_, really) to me: _[The Elements of Style]_, by Strunk and White \[[pdf]\].

_The Elements of Style_ (TEOS), first published in 1935, is one of my favourite non-fiction books of all time. It has improved my writing (such as it is!) immensely. I can't recommend it highly enough. At this point it should come as no surprise to the reader that TEOS is highly opinionated, prescriptive, and _constraining_. It begins by authoritatively enumerating several _hard_ grammatical and syntactic rules that one should expect to be common knowledge for all English speakers. One such basic rule that is _frequently_ abused by native English speakers is

> __#5 - Do not join independent clauses with a comma.__
>
> If two or more clauses grammatically complete and not joined by a conjunction are to form a single compound sentence, the proper mark of punctuation is a semicolon.

(Many people often write sentences of the form _"It is nearly half past five, we cannot reach town before dark"_; this is incorrect, and one should instead write _"It is nearly half past five; we cannot reach town before dark"_. If one wishes to use a comma, then a conjunction is necessary: _"It is nearly half past five, __and/so__ we cannot reach town before dark."_)

The book then follows with a concise and comprehensive treatise on the _design and structure_ of pieces of English writing (the analogy continues!). I have found that adhering to their constraints has freed up my writing -- instead of agonizing over form, style, and structure, I am (generally -- although not always) able to abstract most of these considerations away and just concentrate on what matters: expressing my thoughts and ideas. Just like many C++ programmers employ style guides to constrain themselves to using certain idioms or subsets of the language, TEOS guides us in how to write concise and readable English.

In drawing this connection between writing and programming, I wanted to explore this general and powerful concept: that constraining the output space in certain ways can greatly enhance one's creativity, productivity, and expressiveness. I believe that this concept is well understood (and frequently used to great advantage) in certain disciplines; visual art and music are obvious examples. There's more to think about here, and I'm sure that others have explored these ideas before. Perhaps I'll follow up with another blog post on this topic once I've thought/read about it a bit more.

[Go]: https://golang.org/
[Karma Wiki]: https://karma.wiki
[AIXIjs]: https://github.com/aslanides/aixijs
[Rob Pike]: https://en.wikipedia.org/wiki/Rob_Pike
[commits]: https://github.com/aslanides/aixijs/commits/master
[Honours]: https://physics.anu.edu.au/education/honours/honours_structure.php
[C. M. Savage]: http://people.physics.anu.edu.au/~cms130/
[J. D. Close]: https://researchers.anu.edu.au/researchers/close-jd
[The Elements of Style]: https://en.wikipedia.org/wiki/The_Elements_of_Style
[pdf]: http://www.jlakes.org/ch/web/The-elements-of-style.pdf
