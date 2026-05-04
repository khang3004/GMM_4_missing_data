# Makefile for generating Manim animations

.PHONY: all scene1 scene2 scene3 scene4 clean

# Default quality (can be overridden, e.g., make all QUALITY=h)
# l = low (480p15), m = medium (720p30), h = high (1080p60), k = 4k (2160p60)
QUALITY ?= h

all: intro scene1 scene2 scene3 scene4 outro

intro:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py IntroScene

scene1:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py IllusionOfImputationScene

scene2:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py TwoHiddenLayersScene

scene3:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py PrecisionWeightedUpdateScene

scene4:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py AlternatingCycleScene

outro:
	.venv/bin/manim -p -q$(QUALITY) src/manim/gmm_missing_animations.py OutroScene

clean:
	rm -rf media/
