import time
import multiprocessing

def run_visualizer_process(data_queue):
    """
    Separate process to render Pygame window for Slow Thinking status and ASR.
    """
    try:
        import pygame
    except ImportError:
        print("[Visualizer] Pygame not installed. Visualizer disabled.")
        return

    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Robot Internal State Monitor")

    # Fonts
    def get_font(size):
        # Priority list for Chinese-supporting fonts
        chn_fonts = ['wenquanyimicrohei', 'wqy-microhei', 'simhei', 'microsoftyahei', 'notosanscjksc', 'droidsansfallback']
        for name in chn_fonts:
            path = pygame.font.match_font(name)
            if path:
                try:
                    return pygame.font.Font(path, size)
                except:
                    continue
        # Fallback to system default
        return pygame.font.SysFont("arial", size)

    try:
        font_large = get_font(32)
        font_medium = get_font(20)
        font_small = get_font(16)
    except:
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()
    running = True

    # State
    state = {
        "score": 0,
        "prefill": "Waiting...",
        "insertion": "",
        "locomotion": [],
        "user_text": "",
        "agent_text": "",
        "last_update_slow": 0,
        "last_update_asr": 0
    }

    def wrap_text(text, font, max_width):
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] < max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        return lines

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Non-blocking queue read
        while not data_queue.empty():
            try:
                msg = data_queue.get_nowait()
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "slow_thinking":
                    data = msg.get("data", {})
                    dia = data.get("dialogue")
                    if dia:
                        # state["score"] = dia.speak_now_score
                        state["prefill"] = dia["prefill_text"]
                        state["insertion"] = dia["insertion_point"]

                    loco = data.get("locomotion")
                    if loco is not None:
                        state["locomotion"] = loco
                    state["last_update_slow"] = time.time()

                elif msg_type == "asr":
                    # Expecting {"user": "...", "agent": "..."}
                    data = msg.get("data", {})
                    if "user" in data:
                        state["user_text"] = data["user"]
                    if "agent" in data:
                        state["agent_text"] = data["agent"]
                    state["last_update_asr"] = time.time()

            except Exception:
                pass

        # Drawing
        screen.fill((30, 30, 30)) # Dark background

        # --- SECTION 1: Slow Thinking ---
        # Header
        pygame.draw.rect(screen, (50, 50, 60), (0, 0, width, 300))

        # Speak Score
        current_score = state["score"]
        score_color = (200,110,110) if current_score >= 7 else (200,200,200)
        status_text = ''

        # title_surf = font_medium.render(f"Speak Now Score: {current_score}/10", True, score_color)
        # screen.blit(title_surf, (20, 20))

        status_surf = font_small.render(status_text, True, score_color)
        screen.blit(status_surf, (300, 25))

        # Locomotion
        # loco_y = 60
        # loco_lbl = font_medium.render("Locomotion:", True, (200, 200, 200))
        # screen.blit(loco_lbl, (20, loco_y))

        # if not state["locomotion"]:
        #     loco_text = "None"
        #     loco_surf = font_medium.render(loco_text, True, (150, 150, 150))
        #     screen.blit(loco_surf, (160, loco_y))
        # else:
        #     action_strs = [f"{l.action}({l.parameter})" for l in state["locomotion"]]
        #     full_loco_str = ", ".join(action_strs)
        #     loco_surf = font_medium.render(full_loco_str, True, (100, 200, 255))
        #     screen.blit(loco_surf, (160, loco_y))

        # Prefill
        prefill_y = 110
        lbl_surf = font_medium.render("Prefill Suggestion:", True, (200, 200, 200))
        screen.blit(lbl_surf, (20, prefill_y))

        ins_surf = font_small.render(f"({state['insertion']})", True, (150, 150, 150))
        screen.blit(ins_surf, (220, prefill_y + 5))

        lines = wrap_text(state["prefill"], font_medium, width - 40)
        y_offset = prefill_y + 40
        for line in lines:
            if y_offset > 280: break
            txt_surf = font_medium.render(line, True, (255, 255, 255))
            screen.blit(txt_surf, (20, y_offset))
            y_offset += 30

        # --- SECTION 2: ASR Transcript ---
        pygame.draw.line(screen, (100, 100, 100), (0, 300), (width, 300), 2)

        asr_y = 320
        asr_lbl = font_medium.render("Live Transcript (ASR):", True, (200, 200, 200))
        screen.blit(asr_lbl, (20, asr_y))

        # User
        user_y = 360
        user_lbl = font_small.render("USER:", True, (100, 255, 100))
        screen.blit(user_lbl, (20, user_y))

        user_lines = wrap_text(state["user_text"], font_medium, width - 40)
        u_offset = user_y + 25
        for line in user_lines[:3]: # Show max 3 lines
            txt_surf = font_small.render(line, True, (255, 255, 255))
            screen.blit(txt_surf, (20, u_offset))
            u_offset += 30

        # Agent
        agent_y = 480
        agent_lbl = font_small.render("AGENT:", True, (100, 100, 255))
        screen.blit(agent_lbl, (20, agent_y))

        agent_lines = wrap_text(state["agent_text"], font_medium, width - 40)
        a_offset = agent_y + 25
        for line in agent_lines[:3]:
            txt_surf = font_small.render(line, True, (255, 255, 255))
            screen.blit(txt_surf, (20, a_offset))
            a_offset += 30

        # Footer
        elapsed_slow = time.time() - state["last_update_slow"] if state["last_update_slow"] else 999
        elapsed_asr = time.time() - state["last_update_asr"] if state["last_update_asr"] else 999

        footer_text = f"Last SlowThink: {elapsed_slow:.1f}s | Last ASR: {elapsed_asr:.1f}s"
        time_surf = font_small.render(footer_text, True, (100, 100, 100))
        screen.blit(time_surf, (20, height - 30))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
