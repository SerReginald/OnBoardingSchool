import warnings

warnings.filterwarnings("ignore")

LIBRARY_NAMES = {
    "graphviz": "GraphViz",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "imageio": "Imageio",
    "tensorflow": "TensorFlow",
	"scipy": "SciPy",
    "sklearn": "SkLearn",
    "keras_tuner": "KerasTuner"
}

for lib, name in LIBRARY_NAMES.items():
    try:
        x = __import__(lib)
        print(f"{name}: {x.__version__}")

    except ImportError:
        print(f"⚠️ {name} could not be imported")



