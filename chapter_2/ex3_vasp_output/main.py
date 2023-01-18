
# Read text input
filename = "__files/OUTCAR"
with open(filename, "r") as f:
    lines = f.readlines()
# remove whitespace
for i,l in enumerate(lines):
    lines[i]=l.strip()
# print lines with loop
N=1
max_cpu = 0
for l in lines:
    if l.startswith("LOOP:"):
        print(f"Iteration {N}:")

        new_cpu=float(l.split(":")[1].split()[2])
        if new_cpu > max_cpu:
            Nmax = N
            max_cpu = new_cpu
        N += 1
        print(f"== {l}")
    if l.startswith("energy without entropy"):
        print(f"== {l}")
    if "k-points in BZ" in l:
        storeline=l

print(f"\n\nMaximal cpu time in iteration {Nmax}: {max_cpu}")
print(storeline)
print()


