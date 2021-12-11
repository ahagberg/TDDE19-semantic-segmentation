import sys


def latex_table_head():
    print("\n")
    print("\\begin{table}[H]")
    print("\t\\centering")
    print("\t\\begin{tabular}{c|c|c|c|c}")

def latex_table_foot():
    print("\t\t\\end{tabular}")
    print("\t\\caption{Caption}")
    print("\t\\label{tab:my_label}")
    print("\\end{table}")
    print("\n")

def latex_table_row(cells):
    print("\t\t\t" + " & ".join(cells) + " \\\\")

def latex_bold(s):
    return "\\textbf{" + s + "}"

if __name__ == "__main__":

    filename = sys.argv[1] # Look at segnet_result.txt

    print("Open file '%s'" % filename)

    f = open(filename)

    lines = f.readlines()
    #print(lines)

    latex_table_head()

    for line in lines:
        tokens = line.split()
        #print(tokens)

        if len(tokens) == 0:
            pass
        elif tokens[0] == "precision":
            latex_table_row(["class"] + tokens)
            print("\t\t\t\\hline")
        elif tokens[0] in [str(c) for c in range(100)]:
            latex_table_row(tokens)
        elif tokens[0] == "accuracy":
            print("\t\t\t\\hline")
            latex_table_row([latex_bold(tokens[0])] + ["", ""] + [latex_bold(tokens[1])] + tokens[2:])
        elif tokens[0] in ["macro", "weighted"]:
            latex_table_row([tokens[0] + " " + tokens[1]] + tokens[2:])
        else:
            pass

    latex_table_foot()
