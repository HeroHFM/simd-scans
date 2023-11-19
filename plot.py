from matplotlib import pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":

	with open(sys.argv[1], "r") as file:
		lines = file.readlines()

	x = np.asarray(list(map(int, lines[0].split(", "))))
	
	ys = {}
	for line in lines[1:]:
		line = line.split(", ")
		ys[line[0]] = np.asarray(list(map(float, line[1:])))

	f, ax = plt.subplots(1)
	
	for algo, y in ys.items():
		if "sequential" in algo: sty = "dashed"
		if "binary"     in algo: sty = "dotted"
		if "hybrid"     in algo: sty = "dashdot"

		ax.plot(x, y / 1e6, label=algo, linestyle=sty)

	
	if len(sys.argv) == 3:
		ref_name = sys.argv[2]

		ref = ys[ref_name]

		print("####### START REPORT")
		for k, v in ys.items():
			if k != ref_name:
				diffs = ref / v
				print(f"{k}: min: {np.min(diffs):.3f} avg: {np.mean(diffs):.3f} max: {np.max(diffs):.3f}")
		print("####### END REPORT")


	plt.ylabel("10K random search (milliseconds)")
	plt.xlabel("Number of keys per node")
	plt.legend()
	plt.tight_layout()
	plt.show()
