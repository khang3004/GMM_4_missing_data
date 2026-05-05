# Makefile for generating Manim animations

.PHONY: all scene1 scene2 scene3 scene4 clean

# Default quality (can be overridden, e.g., make all QUALITY=h)
# l = low (480p15), m = medium (720p30), h = high (1080p60), k = 4k (2160p60)
QUALITY ?= h

all: intro scene1 scene2 scene3 scene4 outro

# Quality mapping for directory names
ifeq ($(QUALITY),l)
    RES = 480p15
else ifeq ($(QUALITY),m)
    RES = 720p30
else ifeq ($(QUALITY),h)
    RES = 1080p60
else ifeq ($(QUALITY),k)
    RES = 2160p60
endif

VIDEO_DIR = media/videos/gmm_missing_animations/$(RES)
SCENES = IntroScene IllusionOfImputationScene TwoHiddenLayersScene PrecisionWeightedUpdateScene AlternatingCycleScene OutroScene

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

merge:
	@echo "Nối các cảnh thành video hoàn chỉnh..."
	@cd $(VIDEO_DIR) && printf "file '%s.mp4'\n" $(SCENES) > list.txt
	@cd $(VIDEO_DIR) && ffmpeg -y -f concat -safe 0 -i list.txt -c copy GMM_Missing_Data_Full.mp4
	@rm $(VIDEO_DIR)/list.txt
	@echo "Đã xong! Video tại: $(VIDEO_DIR)/GMM_Missing_Data_Full.mp4"

play:
	open $(VIDEO_DIR)/GMM_Missing_Data_Full.mp4

clean:
	rm -rf media/
