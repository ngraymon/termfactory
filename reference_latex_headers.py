
#
#
#
#
#
#
#
#
#
#
#
w_equations_latex_header = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{physics}
\usepackage{simplewick}

\geometry{legalpaper, landscape, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}

\newcommand{\bh}{\textbf{h}}
\newcommand{\bs}{\textbf{s}}
\newcommand{\bt}{\textbf{t}}
\newcommand{\bw}{\textbf{w}}
\newcommand{\bW}{\textbf{W}}
\newcommand{\bc}{\textbf{c}}
\newcommand{\bd}{\textbf{d}}

\newcommand{\up}[1]{\hat{#1}^{\dagger}}
\newcommand{\down}[1]{\hat{#1}}

\begin{document}
"""
#
#
#
#
#
#
#
#
#
#
#
full_vecc_latex_header = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{physics}
\usepackage{simplewick}

\geometry{legalpaper, landscape, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}

\newcommand{\bh}{\textbf{h}}
\newcommand{\bs}{\textbf{s}}
\newcommand{\bt}{\textbf{t}}
\newcommand{\bw}{\textbf{w}}
\newcommand{\bc}{\textbf{c}}
\newcommand{\bd}{\textbf{d}}

\newcommand{\up}[1]{\hat{#1}^{\dagger}}
\newcommand{\down}[1]{\hat{#1}}

\begin{document}

%
%
%
%
\section{Automated equations}
All equations are derived assuming a Hamiltonian and $\hat{S}$ of the form:
\begin{equation}
    \hat{H} = \sum_{a,b} \ket{a}\bra{b}h^{a}_{b}(q)
\qquad\qquad
    \hat{S} = \sum_{c,d} \ket{c}\bra{d}s^{c}_{d}(q)
\qquad\qquad
    \hat{\Omega} = \down{i} + \up{i} + \down{i}\down{j} + \down{i}\up{j} + \up{i}\up{j} + \cdots
\end{equation}
when we limit ourselves to at most 2nd order terms (creation operator: $\down{b}$, annihilation operator: $\up{b}$)
\begin{equation}
    \bh(q) = \bh + \bh_{i}\down{i} + \bh^{i}\up{i} + \bh^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bh_{ij}\down{i}\down{j} + \frac{1}{2}\bh^{ij}\up{i}\up{j}
\end{equation}
and
\begin{equation}
    \bs(q) = \bs_{i}\down{i} + \bs^{i}\up{i} + \bs^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bs_{ij}\down{i}\down{j} + \frac{1}{2}\bs^{ij}\up{i}\up{j}
\end{equation}

Define the following: (not correct but mechanically useful)
\begin{equation}
    f = \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
\qquad
    \bar{f} = \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
\qquad
    (
    \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
    -
    \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
    )
    = (\bar{f} - f) = 1
\qquad
    \bar{f} = 1, f = 0
\end{equation}
and
\begin{equation}
    \bt_{i} \equiv f\bs_{i}
\qquad
    \bt^{i} \equiv \bar{f}\bs^{i}
\qquad
    \bt^{i}_{j} \equiv \bar{f}f\bs^{i}_{j}
\qquad
    \bt^{ij} \equiv \bar{f}^2\bs^{ij}
\qquad
    \bt_{ij} \equiv f^2\bs_{ij}
\qquad
    \text{and so forth ....}
\end{equation}


The amplitude equation is
\begin{align}
    LHS &= RHS
\\
    \mel{a}{\hat{\Omega}_{\lambda}\dv{\tau}e^{\hat{S}} + \hat{\Omega}_{\lambda}e^{\hat{S}}\varepsilon}{b}
    &= \mel{a}{\hat{\Omega}_{\lambda}\hat{H}e^{\hat{S}}}{b}.
\end{align}

\clearpage

%
%
%
%
%
%


"""
#
#
#
#
#
#
#
#
#
#
#
#
ground_state_vecc_latex_header = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{physics}
\usepackage{simplewick}

\geometry{legalpaper, landscape, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}

\newcommand{\bh}{\textbf{h}}
\newcommand{\bs}{\textbf{s}}
\newcommand{\bt}{\textbf{t}}
\newcommand{\bw}{\textbf{w}}
\newcommand{\bc}{\textbf{c}}
\newcommand{\bd}{\textbf{d}}

\newcommand{\up}[1]{\hat{#1}^{\dagger}}
\newcommand{\down}[1]{\hat{#1}}

\begin{document}

%
%
%
%
\section{Automated equations}
All equations are derived assuming a Hamiltonian and $\hat{S}$ of the form:
\begin{equation}
    \hat{H} = \sum_{a,b} \ket{a}\bra{b}h^{a}_{b}(q)
\qquad\qquad
    \hat{S} = \sum_{c,d} \ket{c}\bra{d}s^{c}_{d}(q)
\qquad\qquad
    \hat{\Omega} = \down{i} + \up{i} + \down{i}\down{j} + \down{i}\up{j} + \up{i}\up{j} + \cdots
\end{equation}
when we limit ourselves to at most 2nd order terms (creation operator: $\down{b}$, annihilation operator: $\up{b}$)
\begin{equation}
    \bh(q) = \bh + \bh_{i}\down{i} + \frac{1}{2}\bh_{ij}\down{i}\down{j}
\end{equation}
and
\begin{equation}
    \bs(q) = \bs^{i}\up{i} + \frac{1}{2}\bs^{ij}\up{i}\up{j}
\end{equation}

Define the following: (not correct but mechanically useful)
\begin{equation}
    f = \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
\qquad
    \bar{f} = \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
\qquad
    (
    \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
    -
    \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
    )
    = (\bar{f} - f) = 1
\qquad
    \bar{f} = 1, f = 0
\end{equation}
and
\begin{equation}
    \bt_{i} \equiv f\bs_{i}
\qquad
    \bt^{i} \equiv \bar{f}\bs^{i}
\qquad
    \bt^{i}_{j} \equiv \bar{f}f\bs^{i}_{j}
\qquad
    \bt^{ij} \equiv \bar{f}^2\bs^{ij}
\qquad
    \bt_{ij} \equiv f^2\bs_{ij}
\qquad
    \text{and so forth ....}
\end{equation}

The amplitude equation is
\begin{align}
    LHS &= RHS
\\
    \mel{a}{\hat{\Omega}_{\lambda}\dv{\tau}e^{\hat{S}} + \hat{\Omega}_{\lambda}e^{\hat{S}}\varepsilon}{b}
    &= \mel{a}{\hat{\Omega}_{\lambda}\hat{H}e^{\hat{S}}}{b}.
\end{align}
\clearpage

%
%
%
%
%
%


"""
#
#
#
#
#
#
#
#
#
#
#
#
full_z_t_symmetric_latex_header = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{physics}
\usepackage{simplewick}

\usepackage{color}
\NewDocumentCommand{\disconnected}{G{}}{\colorbox{yellow}{$#1$}}
\NewDocumentCommand{\red}{G{}}{{\color{red}#1}}
\NewDocumentCommand{\blue}{G{}}{{\color{blue}#1}}

% \geometry{portrait, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}
\geometry{legalpaper, landscape, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}

\newcommand{\bh}{\textbf{h}}
\newcommand{\bt}{\textbf{t}}
\newcommand{\bw}{\textbf{w}}
\newcommand{\bc}{\textbf{c}}
\newcommand{\bd}{\textbf{d}}

\newcommand{\bs}{\textbf{s}}
\newcommand{\bz}{\textbf{z}}
\newcommand{\bS}{\hat{\textbf{S}}_{\gamma}}
\newcommand{\bZ}{\hat{\textbf{Z}}_{\gamma}}
\newcommand{\ebS}{e^{\bS}}
\newcommand{\ebZ}{e^{\bZ}}
\newcommand{\g}{\hat{g}_{\gamma}}
\newcommand{\bg}{\hat{\textbf{g}}_{\gamma}}
\newcommand{\bG}{\textbf{G}}

\newcommand{\up}[1]{\hat{#1}^{\dagger}}
\newcommand{\down}[1]{\hat{#1}}

\begin{document}


%
%
%
%
\section{Automated equations}
All equations are derived assuming a Hamiltonian, $\hat{S}$ and $\hat{Z}$ of the form:
\begin{equation}
    \hat{H} = \sum_{a,b} \ket{a}\bra{b}h^{a}_{b}(q)
\qquad\qquad
    \hat{S} = \sum_{c,d} \ket{c}\bra{d}s^{c}_{d}(q)
\qquad\qquad
    \hat{Z} = \sum_{e>f} \ket{e}\bra{f}z^{e}_{f}(q)
\qquad\qquad
    \hat{\Omega} = \down{i} + \up{i} + \down{i}\down{j} + \down{i}\up{j} + \up{i}\up{j} + \cdots
\end{equation}
when we limit ourselves to at most 2nd order terms (creation operator: $\down{b}$, annihilation operator: $\up{b}$)
\begin{equation}
    \bh(q) = \bh + \bh_{i}\down{i} + \bh^{i}\up{i} + \bh^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bh_{ij}\down{i}\down{j} + \frac{1}{2}\bh^{ij}\up{i}\up{j}
\end{equation}
\begin{equation}
    \bs(q) = \bs_{i}\down{i} + \bs^{i}\up{i} + \bs^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bs_{ij}\down{i}\down{j} + \frac{1}{2}\bs^{ij}\up{i}\up{j}
\end{equation}
\begin{equation}
    \bz(q) = \bz_{0} + \bz_{i}\down{i} + \bz^{i}\up{i} + \bz^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bz_{ij}\down{i}\down{j} + \frac{1}{2}\bz^{ij}\up{i}\up{j}
\end{equation}

Define the following: (not correct but mechanically useful)
\begin{equation}
    f = \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
\qquad
    \bar{f} = \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
\qquad
    (
    \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
    -
    \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
    )
    = (\bar{f} - f) = 1
\qquad
    \bar{f} = 1, f = 0
\end{equation}
and
\begin{equation}
    \bt_{i} \equiv f\bs_{i}
\qquad
    \bt^{i} \equiv \bar{f}\bs^{i}
\qquad
    \bt^{i}_{j} \equiv \bar{f}f\bs^{i}_{j}
\qquad
    \bt^{ij} \equiv \bar{f}^2\bs^{ij}
\qquad
    \bt_{ij} \equiv f^2\bs_{ij}
\qquad
    \text{and so forth ....}
\end{equation}


The ansatz equation
\begin{equation}
    \ket{\psi_{\gamma}(\tau)}
    = e^{i\hat{H}\tau}\ket{\gamma,0}
%
    \approx e^{\hat{Z}_{\gamma}}e^{\hat{S}_{\gamma}}\ket{\gamma,0}
%
    = (\hat{1}+\hat{Z})e^{\hat{S}_{\gamma}}\ket{\gamma,0}
\end{equation}
\begin{equation}
    \ket{\Psi(\tau)} = \sum_{\gamma}\bm{\chi}_\gamma\ket{\psi_{\gamma}(\tau)}
    % \quad (\bm{\chi} \text{ is a vector which is A dimensionality})
\end{equation}

The old amplitude equation is
\begin{align}
    LHS &= RHS
\\
    \mel{a}{\hat{\Omega}_{\lambda}\dv{\tau}e^{\hat{S}} + \hat{\Omega}_{\lambda}e^{\hat{S}}\varepsilon}{b}
    &= \mel{a}{\hat{\Omega}_{\lambda}\hat{H}e^{\hat{S}}}{b}.
\end{align}

The new amplitude equation is
\begin{align}
    LHS &= RHS
\\
    i\Bigg[\dv{\tau}e^{\hat{S}_{\gamma}} + e^{\hat{S}_{\gamma}}\dv{\tau}\hat{Z}_{\gamma}\Bigg]
    &= (\hat{1} - \hat{Z}_{\gamma})\hat{H}(\hat{1} + \hat{Z}_{\gamma})e^{\hat{S}_{\gamma}}
    = \hat{g}e^{\hat{S}_{\gamma}}
    % \bra{a}
    % i\Bigg[\dv{\tau}e^{\hat{S}_{\gamma}} + e^{\hat{S}_{\gamma}}\dv{\tau}\hat{Z}\Bigg]
    % \ket{b}
    % &=
    % \bra{a}
    % \hat{g}e^{\hat{S}_{\gamma}}
    % \ket{b}
\end{align}

so we can project
\begin{align}
    \bra{a}
    % i\hat{\Omega}_{\lambda}\dv{\bS}{\tau}
    i\Bigg[\Omega_{\lambda}\dv{\tau}e^{\hat{S}_{\gamma}} + \Omega_{\lambda}e^{\hat{S}_{\gamma}}\dv{\tau}\hat{Z}_{\gamma}\Bigg]
    \ket{b}
    &=
    \bra{a}
    % \hat{\Omega}_{\lambda}\Big(\bg  - i\dv{\bZ}{\tau}\Big)e^{\bS}
    \Omega_{\lambda}\g e^{\hat{S}_{\gamma}}
    \ket{b}
\end{align}


We have the relation that
\begin{align}
    ie^{\bS}
    \Big(
        \dv{\bS}{\tau} + \dv{\bZ}{\tau}
    \Big)
    &= \bg e^{\bS}
\\
    ie^{\bS}\dv{\bS}{\tau}
    &= \bg e^{\bS} - i\dv{\bZ}{\tau}e^{\bS}
    = \Big(\bg  - i\dv{\bZ}{\tau}\Big)e^{\bS}
\\
    i\dv{\bS}{\tau}
    &= e^{-\bS}\Big(\bg  - i\dv{\bZ}{\tau}\Big)e^{\bS}
\end{align}

\clearpage
%
%
%
%
%
%



"""
#
#
#
#
#
#
#
#
#
#
#
#
ground_state_z_t_symmetric_latex_header = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{physics}
\usepackage{simplewick}

\usepackage{color}
\NewDocumentCommand{\disconnected}{G{}}{\colorbox{yellow}{$#1$}}
\NewDocumentCommand{\red}{G{}}{{\color{red}#1}}
\NewDocumentCommand{\blue}{G{}}{{\color{blue}#1}}

%\geometry{portrait, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}
\geometry{a4paper, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}
%\geometry{legalpaper, landscape, left=0.4in, right=0.4in, top=0.25in, bottom=0.5in}

\newcommand{\bh}{\textbf{h}}
\newcommand{\bt}{\textbf{t}}
\newcommand{\bw}{\textbf{w}}
\newcommand{\bc}{\textbf{c}}
\newcommand{\bd}{\textbf{d}}

\newcommand{\bs}{\textbf{s}}
\newcommand{\bz}{\textbf{z}}
\newcommand{\bS}{\hat{\textbf{S}}_{\gamma}}
\newcommand{\bZ}{\hat{\textbf{Z}}_{\gamma}}
\newcommand{\ebS}{e^{\bS}}
\newcommand{\ebZ}{e^{\bZ}}
\newcommand{\g}{\hat{g}_{\gamma}}
\newcommand{\bg}{\hat{\textbf{g}}_{\gamma}}
\newcommand{\bG}{\textbf{G}}

\newcommand{\up}[1]{\hat{#1}^{\dagger}}
\newcommand{\down}[1]{\hat{#1}}

\begin{document}


%
%
%
%
\section{Automated equations}
All equations are derived assuming a Hamiltonian, $\hat{S}$ and $\hat{Z}$ of the form:
\begin{equation}
    \hat{H} = \sum_{a,b} \ket{a}\bra{b}h^{a}_{b}(q)
\qquad\qquad
    \hat{S} = \sum_{c,d} \ket{c}\bra{d}s^{c}_{d}(q)
\qquad\qquad
    \hat{Z} = \sum_{e>f} \ket{e}\bra{f}z^{e}_{f}(q)
\qquad\qquad
    \hat{\Omega} = \down{i} + \up{i} + \down{i}\down{j} + \down{i}\up{j} + \up{i}\up{j} + \cdots
\end{equation}
when we limit ourselves to at most 2nd order terms (creation operator: $\down{b}$, annihilation operator: $\up{b}$)
\begin{equation}
    \bh(q) = \bh + \bh_{i}\down{i} + \bh^{i}\up{i} + \bh^{i}_{j}\up{i}\down{j} + \frac{1}{2}\bh_{ij}\down{i}\down{j} + \frac{1}{2}\bh^{ij}\up{i}\up{j}
\end{equation}
\begin{equation}
    \bs(q) = \bs_{0} + \bs^{i}\up{i} + \frac{1}{2}\bs^{ij}\up{i}\up{j}
\end{equation}
\begin{equation}
    \bz(q) = \bz_{0} + \bz^{i}\up{i} + \frac{1}{2}\bz^{ij}\up{i}\up{j}
\end{equation}

Define the following: (not correct but mechanically useful)
\begin{equation}
    f = \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
\qquad
    \bar{f} = \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
\qquad
    (
    \contraction[1ex]{}{\down{b}}{}{\up{b}}\down{b}\up{b}
    -
    \contraction[1ex]{}{\up{b}}{}{\down{b}}\up{b}\down{b}
    )
    = (\bar{f} - f) = 1
\qquad
    \bar{f} = 1, f = 0
\end{equation}
and
\begin{equation}
    \bt_{i} \equiv f\bs_{i}
\qquad
    \bt^{i} \equiv \bar{f}\bs^{i}
\qquad
    \bt^{i}_{j} \equiv \bar{f}f\bs^{i}_{j}
\qquad
    \bt^{ij} \equiv \bar{f}^2\bs^{ij}
\qquad
    \bt_{ij} \equiv f^2\bs_{ij}
\qquad
    \text{and so forth ....}
\end{equation}


The ansatz equation
\begin{equation}
    \ket{\psi_{\gamma}(\tau)}
    = e^{i\hat{H}\tau}\ket{\gamma,0}
%
    \approx e^{\hat{S}_{\gamma}}e^{\hat{Z}_{\gamma}}\ket{\gamma,0}
%
    = e^{\hat{S}_{\gamma}}(\hat{1}+\hat{Z})\ket{\gamma,0}
\end{equation}
\begin{equation}
    \ket{\Psi(\tau)} = \sum_{\gamma}\bm{\chi}_\gamma\ket{\psi_{\gamma}(\tau)}
    % \quad (\bm{\chi} \text{ is a vector which is A dimensionality})
\end{equation}

The new amplitude equation is
\begin{align}
    LHS &= RHS
\\
    (\hat{1} - \hat{Z}_{\gamma})\left(i\dv{\hat{S}_{\gamma}}{\tau}\right)(\hat{1} + \hat{Z}_{\gamma})
    &= (\hat{1} - \hat{Z}_{\gamma})e^{-\hat{S}_{\gamma}}\hat{H}e^{\hat{S}_{\gamma}}(\hat{1} + \hat{Z}_{\gamma})
%
\\  &= (\hat{1} - \hat{Z}_{\gamma})\bar{H}_{\gamma}(\hat{1} + \hat{Z}_{\gamma})
\\  &= \hat{G}
\end{align}

where
\begin{align}
    i\dv{\hat{S}_{\gamma}}{\tau} = \left(\bar{H} + \bar{H}\hat{Z}\right)_{\gamma\gamma} = \hat{G}_{\gamma\gamma}
\end{align}

and

\begin{align}
    i\dv{\hat{Z}_{x\gamma}}{\tau} = \left(\bar{H} + \bar{H}\hat{Z}\right)_{x\gamma} - i\dv{\hat{S}_{\gamma}}{\tau}\hat{Z}_{x\gamma}
\end{align}

\clearpage
%
%
%
%
%
%


"""
#
#
#
#
#
#
#
#
#
#
#
#
