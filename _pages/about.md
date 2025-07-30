---
permalink: /
title: "Separable Physics-Informed Neural Networks"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---


Welcome to my blog post describing and explaining the paper "Separable Physics-Informed Neural Networks" by Cho et al., which was presented at NeurIPS 2023. This research describes a unique and scalable method for solving PDEs that restructures how neural networks model multidimensional input. As someone who is very interested in the convergence of deep learning and computational physics, I found SPINNs to be a novel and interesting approach to addressing the bottlenecks of previous PINN structures. It demonstrates how traditional concepts such as variable separation and tensor decomposition may be combined with neural networks to provide faster, more efficient scientific computing models.

This blog article is based on my reading of the research, my seminar presentation on SPINNs, and comparisons with other comparable methods such as PINNs, Causal PINNs, and low-rank neural PDE solvers. I've endeavored to make the concepts comprehensible even if you're not deeply into numerical methods or theoretical machine learning.

In this blog post, we examine Separable Physics-Informed Neural Networks (SPINNs), a sophisticated method for applying machine learning to the solution of high-dimensional partial differential equations (PDEs). The exponential cost of sampling and differentiation in high dimensions affects the scalability of traditional Physics-Informed Neural Networks (PINNs). By segmenting the network according to input dimensions and utilizing forward-mode automatic differentiation, SPINNs provide a straightforward yet effective redesign. When combined, these two concepts and result in significant increases in speed and memory effectiveness. With an emphasis on the physics-inspired motivations and little mathematical overhead, this post seeks to make the concepts underlying SPINNs understandable and approachable. Let’s decompose.


INTRODUCTION
======

Partial differential equations (PDEs) are a fundamental problem in computational science and engineering. PDEs are used in a variety of applications, including fluid flow and electromagnetic field simulations, as well as modeling quantum systems and weather patterns.  
Traditional methods to PDE solutions are Finite Element Methods or Finite Difference Methods which require lots of computation and it struggle with high dimensions. SO it was so challenging for In real-world scenarios, especially in high dimensions or complex geometries, classical solvers become too expensive or impractical.

In recent years, Physics-Informed Neural Networks (PINNs) have gained popularity as a mesh-free, data-efficient alternative. PINNs learn PDE solutions by explicitly encoding physical rules into a neural network's loss function. They operate exceptionally well for many situations, particularly in low dimensions, and require no labeled data. . Despite its potential, PINNs encounter significant challenges when used to high-dimensional PDEs. The number of points required to adequately record the physics explodes, and computing gradients using reverse-mode automated differentiation becomes prohibitively expensive. 

This is where the study "Separable Physics-Informed Neural Networks" comes in. SPINNs address the fundamental constraints of PINNs by making two innovative modifications: 
1. A separable architecture uses small MLPs for each input dimension and a low-rank tensor product to assemble the full solution. 
2. Forward-mode automatic differentiation is ideal for computing multiple output derivatives with few inputs, a common scenario in physics problems.
These changes may appear modest, but their influence is profound: SPINNs can achieve up to 62× speedups, reduce memory utilization by 29×, and handle PDEs in up to (3+1)D utilizing commodity GPUs, all while maintaining or enhancing accuracy.



What Are PINNs and Where Do They Struggle? 
======


A PINN is a neural network that learns a function u(x) satisfying a PDE by minimizing the residuals of the PDE and initial/boundary conditions using automatic differentiation (AD).
PINNs approximate the solution u(x) to a PDE by minimizing 

• Residual loss using differential equation.

• Losses due to initial and boundary conditions. 

These are enforced using automatic differentiation (often reverse-mode) and trained with standard gradient descent. These models are mesh-free, data-efficient (unsupervised), and can solve both forward and inverse problems. 

However, the cost of analyzing the network and computing gradients rises considerably as dimensionality increases. To solve PDEs with finer grids or higher dimensions, PINNs require 

• a large number of collocation points. 

• Cannot handle large training sets on single gpu

• they face high computational and memory costs 

• they suffer from slow convergence.

These scalability limits are well-documented [1, 3]. SPINNs confront things head on. 


Introducing SPINNS
======

So to overcome all the problems researchers comes with a solution i.e SPINNS- which stands for Separable Physics-Informed Neural Networks.
It's a new way to structure and train PINNs that:

•	Handles multi-dimensional PDEs more efficiently

•	Allows using more collocation points (>10 million!) even on a single GPU

•	Is much faster and more accurate than traditional PINNs

To understand why SPINN is so quick, we must first consider how derivatives are produced in deep learning – especially, by automatic differentiation. There are two major modes:

Reverse-Mode AD
Reverse-Mode AD (used in backpropagation) is ideal for functions with multiple inputs and a single output (e.g., loss functions). It operates by first performing a forward pass to compute outputs, followed by a backward pass to calculate gradients with respect to inputs. This is useful for training neural networks that require the gradient of a scalar loss with respect to all model parameters.
Forward-Mode AD:
Forward-Mode AD works best for functions with few inputs and many outputs, such as computing derivatives of solutions with respect to input coordinates in PDEs. Forward-mode AD computes these derivatives immediately during the forward pass, making it more efficient when assessing PDE residuals.

In SPINN, the number of input variables (coordinates) is minimal (for example, time and space), but the number of outputs (solution evaluations at various points) is huge, making forward-mode AD the best fit.



SPINN Architecture
======












Many of the features of dynamic content management systems (like Wordpress) can be achieved in this fashion, using a fraction of the computational resources and with far less vulnerability to hacking and DDoSing. You can also modify the theme to your heart's content without touching the content of your site. If you get to a point where you've broken something in Jekyll/HTML/CSS beyond repair, your Markdown files describing your talks, publications, etc. are safe. You can rollback the changes or even delete the repository and start over - just be sure to save the Markdown files! You can also write scripts that process the structured data on the site, such as [this one](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb) that analyzes metadata in pages about talks to display [a map of every location you've given a talk](https://academicpages.github.io/talkmap.html).

For those users that need more advanced functionality, the template also supports the following popular tools:
- [MathJax](https://www.mathjax.org/) for mathematical equations
- [Mermaid](https://mermaid.js.org/) for diagraming
- [Plotly](https://plotly.com/javascript/) for plotting

Getting started
======
1. Register a GitHub account if you don't have one and confirm your e-mail (required!)
1. Fork [this template](https://github.com/academicpages/academicpages.github.io) by clicking the "Use this template" button in the top right. 
1. Go to the repository's settings (rightmost item in the tabs that start with "Code", should be below "Unwatch"). Rename the repository "[your GitHub username].github.io", which will also be your website's URL.
1. Set site-wide configuration and create content & metadata (see below -- also see [this set of diffs](http://archive.is/3TPas) showing what files were changed to set up [an example site](https://getorg-testacct.github.io) for a user with the username "getorg-testacct")
1. Upload any files (like PDFs, .zip files, etc.) to the files/ directory. They will appear at https://[your GitHub username].github.io/files/example.pdf.  
1. Check status by going to the repository settings, in the "GitHub pages" section

Site-wide configuration
------
The main configuration file for the site is in the base directory in [_config.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_config.yml), which defines the content in the sidebars and other site-wide features. You will need to replace the default variables with ones about yourself and your site's github repository. The configuration file for the top menu is in [_data/navigation.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_data/navigation.yml). For example, if you don't have a portfolio or blog posts, you can remove those items from that navigation.yml file to remove them from the header. 

Create content & metadata
------
For site content, there is one Markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a Markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each Markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

The repository includes [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual Markdown files that will be properly formatted for the Academic Pages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the Markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and Markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a Markdown file for a talk
![Editing a Markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring Academic Pages can be found in [the guide](https://academicpages.github.io/markdown/), the [growing wiki](https://github.com/academicpages/academicpages.github.io/wiki), and you can always [ask a question on GitHub](https://github.com/academicpages/academicpages.github.io/discussions). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
