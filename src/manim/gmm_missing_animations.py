from manim import *
import numpy as np

# Theme colors
TECH_BLUE = "#0056A0"
ALERT_RED = "#D32F2F"
MISSING_COLOR = GRAY

config.background_color = "#1C1C1C"

# Helper function to generate cluster data
def get_cluster_dots(axes, mean, cov, n_points, color, seed=42):
    np.random.seed(seed)
    points = np.random.multivariate_normal(mean, cov, n_points)
    dots = VGroup(*[Dot(axes.c2p(p[0], p[1]), color=color) for p in points])
    return dots, points

class IntroScene(Scene):
    def construct(self):
        # YouTube Style Intro & Author Shoutout
        greeting = Tex("Welcome back!", font_size=48, color=TECH_BLUE)
        topic = Tex("Today we're exploring a beautiful idea from the paper:", font_size=36)
        
        paper_title = Tex(r"\textbf{Gaussian Mixture Model Clustering}\\\textbf{with Incomplete Data}", font_size=44, color=YELLOW)
        authors = Tex("By: Khang Nguyen, Nhat Nguyen \\& Phuc Nguyen", font_size=32)
        institute = Tex("HCMUS Data Science Seminar", font_size=28, color=GRAY)
        
        intro_group = VGroup(greeting, topic, paper_title, authors, institute).arrange(DOWN, buff=0.5)
        
        self.play(Write(greeting))
        self.wait(1)
        self.play(FadeIn(topic, shift=UP))
        self.wait(0.5)
        self.play(Write(paper_title))
        self.wait(1)
        self.play(FadeIn(authors), FadeIn(institute))
        self.wait(3)
        self.play(FadeOut(intro_group))


class IllusionOfImputationScene(Scene):
    def construct(self):
        # 1. Setup Axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            axis_config={"color": WHITE, "include_ticks": False}
        )
        self.play(Create(axes), run_time=1.5)

        # 2. Setup Data
        mean1, cov1 = [-2, -2], [[0.5, 0], [0, 0.5]]
        mean2, cov2 = [2, 2], [[0.5, 0], [0, 0.5]]
        
        dots1, pts1 = get_cluster_dots(axes, mean1, cov1, 30, TECH_BLUE, seed=1)
        dots2, pts2 = get_cluster_dots(axes, mean2, cov2, 30, ALERT_RED, seed=2)
        
        self.play(FadeIn(dots1, lag_ratio=0.05), FadeIn(dots2, lag_ratio=0.05))
        self.wait(1)

        # 3. Introduce missingness (lose y coordinate for some points)
        missing_indices_1 = [5, 10, 15, 20, 25]
        missing_indices_2 = [3, 8, 12, 18, 22]
        
        missing_dots = VGroup()
        animations = []
        for i in missing_indices_1:
            dots1[i].generate_target()
            dots1[i].target.set_color(MISSING_COLOR)
            dots1[i].target.move_to(axes.c2p(pts1[i][0], 0))
            animations.append(MoveToTarget(dots1[i]))
            missing_dots.add(dots1[i])
            
        for i in missing_indices_2:
            dots2[i].generate_target()
            dots2[i].target.set_color(MISSING_COLOR)
            dots2[i].target.move_to(axes.c2p(pts2[i][0], 0))
            animations.append(MoveToTarget(dots2[i]))
            missing_dots.add(dots2[i])

        missing_text = Tex("Data goes missing...", font_size=36).to_edge(UP)
        self.play(Write(missing_text))
        self.play(*animations, run_time=2)
        self.wait(1)

        # 4. Mean Imputation (Wrong approach)
        impute_text = Tex("Naive Mean Imputation", font_size=36, color=YELLOW).to_edge(UP)
        
        # Calculate overall mean y
        all_y = np.concatenate([pts1[:, 1], pts2[:, 1]])
        mean_y = np.mean(all_y)
        
        impute_animations = []
        for dot in missing_dots:
            dot.generate_target()
            # Move to overall mean y, keeping its original x
            current_x = axes.p2c(dot.get_center())[0]
            dot.target.move_to(axes.c2p(current_x, mean_y))
            impute_animations.append(MoveToTarget(dot))
            
        self.play(Transform(missing_text, impute_text))
        self.play(*impute_animations, run_time=2)
        
        # 5. Highlight the destruction of clusters
        destruction_text = Tex(r"Clusters are mixed \& geometry is destroyed", font_size=32, color=RED).next_to(impute_text, DOWN)
        self.play(Write(destruction_text))
        
        # Draw an ellipse around the mixed center area
        mixed_area = Ellipse(width=5, height=1, color=YELLOW).move_to(axes.c2p(0, mean_y))
        self.play(Create(mixed_area))
        self.play(Circumscribe(mixed_area))
        self.wait(2)


class TwoHiddenLayersScene(Scene):
    def construct(self):
        # 1. Start with the standard GMM equation
        title = Tex("Joint Optimization View", font_size=40, color=TECH_BLUE).to_edge(UP)
        self.play(Write(title))

        eq_base = MathTex(
            r"\max_{\theta} \sum_{j=1}^{n} \log \left( \sum_{k=1}^{K} \alpha_k \mathcal{N}(x_j \mid \mu_k, \Sigma_k) \right)",
            font_size=48
        )
        self.play(FadeIn(eq_base))
        self.wait(1)

        # 2. Transform to the missing data formulation
        eq_missing = MathTex(
            r"\max_{X, \theta} \sum_{j=1}^{n} \log \left( \sum_{k=1}^{K} \alpha_k \mathcal{N}(x_j \mid \mu_k, \Sigma_k) \right)",
            font_size=48
        )
        # Highlight X and theta
        eq_missing.set_color_by_tex("X", YELLOW)
        
        text_explain = Tex("Optimize the completed matrix X, not just parameters!", font_size=32, color=YELLOW).next_to(eq_missing, DOWN, buff=1)
        
        self.play(TransformMatchingTex(eq_base, eq_missing))
        self.play(Write(text_explain))
        self.wait(2)
        
        self.play(FadeOut(eq_missing), FadeOut(text_explain))

        # 3. Two hidden layers
        layer1_text = Tex("1. Cluster Responsibility: ", font_size=36)
        layer1_math = MathTex(r"\gamma_{jk}", font_size=40, color=TECH_BLUE)
        layer1 = VGroup(layer1_text, layer1_math).arrange(RIGHT)
        
        layer2_text = Tex("2. Missing Features: ", font_size=36)
        layer2_math = MathTex(r"x_{j,m}", font_size=40, color=ALERT_RED)
        layer2 = VGroup(layer2_text, layer2_math).arrange(RIGHT)
        
        layers = VGroup(layer1, layer2).arrange(DOWN, aligned_edge=LEFT, buff=0.8).center()
        
        self.play(Write(layer1))
        self.wait(1)
        self.play(Write(layer2))
        self.wait(2)

        locked_text = Tex("Observed entries are locked. Only missing blocks move.", font_size=32, color=YELLOW).next_to(layers, DOWN, buff=1)
        self.play(Write(locked_text))
        self.wait(2)


class PrecisionWeightedUpdateScene(Scene):
    def construct(self):
        title = Tex("Precision-Weighted Reconstruction", font_size=36, color=TECH_BLUE).to_edge(UP)
        self.play(Write(title))

        # 1. Setup zoom geometry
        axes = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], axis_config={"include_ticks": False})
        
        # Two Gaussian clusters (Ellipses)
        g1 = Ellipse(width=3, height=2, color=TECH_BLUE).move_to(axes.c2p(-1.5, 1.5))
        g1_center = Dot(g1.get_center(), color=TECH_BLUE)
        g1_label = MathTex(r"\mu_1, \Lambda_1").next_to(g1, UP)
        
        g2 = Ellipse(width=2, height=3, color=ALERT_RED).move_to(axes.c2p(1.5, -1.5))
        g2_center = Dot(g2.get_center(), color=ALERT_RED)
        g2_label = MathTex(r"\mu_2, \Lambda_2").next_to(g2, DOWN)
        
        self.play(Create(g1), Create(g1_center), Write(g1_label), 
                  Create(g2), Create(g2_center), Write(g2_label))
        
        # 2. The missing point
        # Known x = 0.5, missing y
        constraint_line = DashedLine(axes.c2p(0.5, -3), axes.c2p(0.5, 3), color=GRAY)
        missing_dot = Dot(axes.c2p(0.5, 0), color=YELLOW, radius=0.1)
        
        self.play(Create(constraint_line))
        self.play(FadeIn(missing_dot))
        
        missing_label = MathTex(r"x_{j,m}").next_to(missing_dot, RIGHT)
        self.play(Write(missing_label))
        self.wait(1)
        
        # 3. Pulling vectors
        # If G1 takes it, it wants it near y=1.5
        target1 = axes.c2p(0.5, 1.2)
        arrow1 = Arrow(missing_dot.get_center(), target1, buff=0, color=TECH_BLUE, stroke_width=4)
        
        # If G2 takes it, it wants it near y=-1.5
        target2 = axes.c2p(0.5, -1.8)
        arrow2 = Arrow(missing_dot.get_center(), target2, buff=0, color=ALERT_RED, stroke_width=4)
        
        self.play(GrowArrow(arrow1))
        self.play(GrowArrow(arrow2))
        
        # 4. Show the equation
        update_eq = MathTex(
            r"x_{j,m}^{\star} = \left(\sum_k \gamma_{jk} \Lambda_{k,mm}\right)^{-1} \sum_k \gamma_{jk} (\text{local proposal})",
            font_size=36
        ).to_edge(DOWN)
        
        self.play(Write(update_eq))
        
        # Move dot to the weighted average (closer to G2 since it's nearer in x)
        final_pos = axes.c2p(0.5, -0.8)
        self.play(
            missing_dot.animate.move_to(final_pos),
            missing_label.animate.next_to(final_pos, RIGHT),
            FadeOut(arrow1), FadeOut(arrow2),
            run_time=2
        )
        self.play(Indicate(missing_dot))
        self.wait(2)


class AlternatingCycleScene(Scene):
    def construct(self):
        title = Tex("The EM Cycle in Action", font_size=36, color=TECH_BLUE).to_edge(UP)
        self.play(Write(title))
        
        # 1. Setup Axes and Data (similar to Scene 1)
        axes = Axes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], axis_config={"include_ticks": False})
        axes.scale(0.8).move_to(LEFT * 2)
        self.play(Create(axes))
        
        # Data
        mean1, cov1 = [-2, -2], [[0.5, 0], [0, 0.5]]
        mean2, cov2 = [2, 2], [[0.5, 0], [0, 0.5]]
        np.random.seed(42)
        pts1 = np.random.multivariate_normal(mean1, cov1, 20)
        pts2 = np.random.multivariate_normal(mean2, cov2, 20)
        
        all_dots = VGroup()
        missing_dots = VGroup()
        
        for p in pts1:
            if np.random.rand() < 0.3: # missing y
                dot = Dot(axes.c2p(p[0], 0), color=GRAY)
                missing_dots.add(dot)
                all_dots.add(dot)
            else:
                all_dots.add(Dot(axes.c2p(p[0], p[1]), color=WHITE))
                
        for p in pts2:
            if np.random.rand() < 0.3: # missing y
                dot = Dot(axes.c2p(p[0], 0), color=GRAY)
                missing_dots.add(dot)
                all_dots.add(dot)
            else:
                all_dots.add(Dot(axes.c2p(p[0], p[1]), color=WHITE))
                
        self.play(FadeIn(all_dots))
        
        # 2. Cycle Flowchart on the right
        flow_group = VGroup()
        e_step = Tex("E-step", font_size=30).move_to(RIGHT * 3 + UP * 1)
        m_params = Tex("M-step (Params)", font_size=30).next_to(e_step, DOWN, buff=1)
        m_data = Tex("M-step (Data)", font_size=30).next_to(m_params, DOWN, buff=1)
        
        flow_group.add(e_step, m_params, m_data)
        
        arrow_e_p = Arrow(e_step.get_bottom(), m_params.get_top(), buff=0.1, stroke_width=2)
        arrow_p_d = Arrow(m_params.get_bottom(), m_data.get_top(), buff=0.1, stroke_width=2)
        arrow_d_e = CurvedArrow(m_data.get_left(), e_step.get_left(), angle=PI/2, color=YELLOW)
        
        self.play(Write(flow_group), Create(arrow_e_p), Create(arrow_p_d), Create(arrow_d_e))
        
        # 3. Simulate Iterations
        g1_ellipse = Ellipse(width=4, height=4, color=TECH_BLUE, fill_opacity=0.1).move_to(axes.c2p(0, 0))
        g2_ellipse = Ellipse(width=4, height=4, color=ALERT_RED, fill_opacity=0.1).move_to(axes.c2p(0, 0))
        self.play(FadeIn(g1_ellipse), FadeIn(g2_ellipse))
        
        for i in range(3):
            # E-step
            self.play(Indicate(e_step, color=YELLOW, scale_factor=1.2), run_time=0.5)
            # Color points based on proximity
            color_anims = []
            for dot in all_dots:
                if dot not in missing_dots:
                    x, y = axes.p2c(dot.get_center())
                    if x < 0:
                        color_anims.append(dot.animate.set_color(TECH_BLUE))
                    else:
                        color_anims.append(dot.animate.set_color(ALERT_RED))
            if color_anims:
                self.play(*color_anims, run_time=0.5)
                
            # M-step params
            self.play(Indicate(m_params, color=YELLOW, scale_factor=1.2), run_time=0.5)
            target_g1_w = 4 - i
            target_g1_h = 4 - i
            target_g2_w = 4 - i
            target_g2_h = 4 - i
            
            target_g1_x = -0.5 * (i+1)
            target_g1_y = -0.5 * (i+1)
            target_g2_x = 0.5 * (i+1)
            target_g2_y = 0.5 * (i+1)
            
            self.play(
                g1_ellipse.animate.stretch_to_fit_width(target_g1_w).stretch_to_fit_height(target_g1_h).move_to(axes.c2p(target_g1_x, target_g1_y)),
                g2_ellipse.animate.stretch_to_fit_width(target_g2_w).stretch_to_fit_height(target_g2_h).move_to(axes.c2p(target_g2_x, target_g2_y)),
                run_time=1
            )
            
            # M-step data
            self.play(Indicate(m_data, color=YELLOW, scale_factor=1.2), run_time=0.5)
            move_anims = []
            for dot in missing_dots:
                x, y = axes.p2c(dot.get_center())
                # Move towards the proper y based on x
                if x < 0:
                    target_y = target_g1_y + (np.random.rand()-0.5)*0.5 # slightly fuzzy
                    dot.generate_target()
                    dot.target.set_color(TECH_BLUE)
                    dot.target.move_to(axes.c2p(x, target_y))
                    move_anims.append(MoveToTarget(dot))
                else:
                    target_y = target_g2_y + (np.random.rand()-0.5)*0.5
                    dot.generate_target()
                    dot.target.set_color(ALERT_RED)
                    dot.target.move_to(axes.c2p(x, target_y))
                    move_anims.append(MoveToTarget(dot))
                    
            if move_anims:
                self.play(*move_anims, run_time=1)
                
        # Final resolution
        self.play(Circumscribe(axes))
        success_text = Tex("Geometry Recovered!", font_size=32, color=GREEN).next_to(axes, UP)
        self.play(Write(success_text))
        self.wait(2)


class OutroScene(Scene):
    def construct(self):
        # YouTube Outro
        thanks = Tex("Thanks for watching!", font_size=48, color=TECH_BLUE)
        sub = Tex(r"If you found this geometric perspective helpful,\\please like and subscribe!", font_size=36)
        
        outro_group = VGroup(thanks, sub).arrange(DOWN, buff=0.5)
        
        self.play(Write(thanks))
        self.wait(1)
        self.play(FadeIn(sub, shift=UP))
        self.wait(3)
        self.play(FadeOut(outro_group))
