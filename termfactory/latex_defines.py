
#  use these if we define \newcommand to map textbf to \bt and so forth
if True:
    bold_t_latex = "\\bt"
    bold_h_latex = "\\bh"
    bold_w_latex = "\\bw"
    bold_c_latex = "\\bc"
    bold_d_latex = "\\bd"
    bold_z_latex = "\\bz"
    bold_s_latex = "\\bs"
    bold_G_latex = "\\bG"
    bold_dt_dtau_latex = "\\dv{\\textbf{t}}{\tau}"
else:
    bold_t_latex = "\\textbf{t}"
    bold_h_latex = "\\textbf{h}"
    bold_w_latex = "\\textbf{w}"
    bold_c_latex = "\\textbf{c}"
    bold_d_latex = "\\textbf{d}"
    bold_z_latex = "\\textbf{z}"
    bold_s_latex = "\\textbf{s}"
    bold_G_latex = "\\textbf{G}"
