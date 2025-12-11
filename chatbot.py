#!/usr/bin/env python3
"""
Chemical Brain Chatbot - A chatbot where memory IS architecture

This is a proof-of-concept chatbot that demonstrates:
- Simulated neurochemicals influencing behavior
- Dynamic neural network that grows and prunes
- Memories tied to chemical states
- Emergent responses from structure

Now using ThreeSystemBrain - the concentrated intelligence architecture!

Run with: python chatbot.py
"""

import os
import sys
import time
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import create_brain, ThreeSystemBrain


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Background
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def make_bar(value: float, width: int = 20, filled: str = '█', empty: str = '░') -> str:
    """Create a progress bar"""
    filled_count = int(value * width)
    return filled * filled_count + empty * (width - filled_count)


def colorize_level(value: float) -> str:
    """Color a value based on its level"""
    if value > 0.7:
        return Colors.RED
    elif value > 0.5:
        return Colors.YELLOW
    elif value > 0.3:
        return Colors.GREEN
    else:
        return Colors.CYAN


def print_dashboard(brain: ThreeSystemBrain) -> None:
    """Print the brain status dashboard"""
    data = brain.get_dashboard_data()

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  THREE-SYSTEM BRAIN STATUS{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    # Mood
    mood = data.get('mood', 'neutral')
    mood_colors = {
        'excited': Colors.YELLOW,
        'happy': Colors.GREEN,
        'calm': Colors.BLUE,
        'curious': Colors.CYAN,
        'focused': Colors.MAGENTA,
        'alert': Colors.YELLOW,
        'tired': Colors.DIM,
        'anxious': Colors.RED,
        'neutral': Colors.WHITE,
    }
    mood_color = mood_colors.get(mood, Colors.WHITE)
    print(f"  Mood: {mood_color}{Colors.BOLD}{mood.upper()}{Colors.RESET}")
    print()

    # Neuromodulators (Chemicals)
    print(f"  {Colors.BOLD}Neuromodulators:{Colors.RESET}")
    chemicals = data.get('chemicals', {})
    for chem, level in chemicals.items():
        color = colorize_level(level)
        bar = make_bar(level)
        print(f"    {chem:15} {color}[{bar}]{Colors.RESET} {level:.2f}")
    print()

    # Network Stats
    print(f"  {Colors.BOLD}Neural Network:{Colors.RESET}")
    neurons = data.get('neurons', {})
    network = data.get('network', {})
    print(f"    Total Neurons:     {neurons.get('total', 0):,}")
    print(f"    Active Neurons:    {neurons.get('active', 0):,}")
    print(f"    Sensory:           {neurons.get('sensory', 0):,}")
    print(f"    Hidden:            {neurons.get('hidden', 0):,}")
    print(f"    Output:            {neurons.get('output', 0):,}")
    print(f"    Sparsity:          {network.get('connection_density', 0.02):.2%}")
    print()

    # Personality
    print(f"  {Colors.BOLD}Personality:{Colors.RESET}")
    personality = data.get('personality', {})
    for trait, value in personality.items():
        bar = make_bar(value, width=15)
        print(f"    {trait:18} [{bar}] {value:.2f}")
    print()

    # Energy
    energy = data.get('energy', 1.0)
    energy_bar = make_bar(energy)
    energy_color = Colors.GREEN if energy > 0.5 else Colors.YELLOW if energy > 0.25 else Colors.RED
    print(f"  Energy: {energy_color}[{energy_bar}]{Colors.RESET} {energy:.2f}")
    print()

    # Regions (if available)
    regions = data.get('regions', {})
    if regions:
        print(f"  {Colors.BOLD}Brain Regions:{Colors.RESET}")
        for region, info in regions.items():
            activity = info.get('activity', 0)
            active = info.get('active_neurons', 0)
            bar = make_bar(min(activity * 10, 1.0), width=10)  # Scale activity
            print(f"    {region:15} [{bar}] active: {active}")
        print()

    print(f"  {Colors.DIM}Step: {brain.state.simulation_step}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def print_response(response: str, result: dict) -> None:
    """Print the brain's response with context"""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}Brain:{Colors.RESET} {response}")

    # Show mood and confidence
    mood = result.get('mood', 'neutral')
    confidence = result.get('confidence', 0.5)
    dopamine = result.get('dopamine', 0.5)
    
    print(f"{Colors.DIM}  [Mood: {mood} | Confidence: {confidence:.2f} | Dopamine: {dopamine:.2f}]{Colors.RESET}")
    print()


def print_help():
    """Print help information"""
    print(f"""
{Colors.BOLD}Integrated Brain Chatbot - Commands{Colors.RESET}

  {Colors.CYAN}/status{Colors.RESET}      - Show detailed brain status dashboard
  {Colors.CYAN}/introspect{Colors.RESET}  - Have the brain describe itself
  {Colors.CYAN}/chemicals{Colors.RESET}   - Show neuromodulator levels
  {Colors.CYAN}/stats{Colors.RESET}       - Show brain statistics
  {Colors.CYAN}/regions{Colors.RESET}     - Show brain region activity
  {Colors.CYAN}/clear{Colors.RESET}       - Clear the screen
  {Colors.CYAN}/save{Colors.RESET}        - Save brain state
  {Colors.CYAN}/help{Colors.RESET}        - Show this help
  {Colors.CYAN}/quit{Colors.RESET}        - Exit the chatbot

Just type naturally to chat with the brain!
""")


def main():
    """Main chatbot loop"""
    print(f"""
{Colors.BOLD}{Colors.CYAN}
+===========================================================+
|                                                           |
|   THREE-SYSTEM BRAIN CHATBOT                              |
|                                                           |
|   Concentrated Intelligence Architecture:                 |
|   - System 1: Sparse Cortical Engine                      |
|   - System 2: Dynamic Recurrent Core                      |
|   - System 3: Neuromodulated Learning                     |
|                                                           |
+===========================================================+
{Colors.RESET}
Type {Colors.CYAN}/help{Colors.RESET} for commands, or just start chatting!
""")

    # Initialize the brain
    print(f"{Colors.DIM}Initializing brain...{Colors.RESET}")
    brain = create_brain('small', use_gpu=False)
    total_neurons = brain.config.total_neurons()
    print(f"{Colors.DIM}Brain ready with {total_neurons:,} neurons.{Colors.RESET}\n")

    while True:
        try:
            # Get input
            user_input = input(f"{Colors.GREEN}You:{Colors.RESET} ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()[0]

                if cmd in ['/quit', '/exit', '/q']:
                    print(f"\n{Colors.CYAN}Brain shutting down after {brain.state.simulation_step} steps.{Colors.RESET}")
                    break

                elif cmd == '/status':
                    print_dashboard(brain)

                elif cmd == '/introspect':
                    data = brain.get_dashboard_data()
                    mood = data.get('mood', 'neutral')
                    neurons = data.get('neurons', {})
                    personality = data.get('personality', {})
                    
                    print(f"\n{Colors.MAGENTA}{Colors.BOLD}Brain Introspection:{Colors.RESET}")
                    print(f"  I am a neural network with {neurons.get('total', 0):,} neurons.")
                    print(f"  Currently {neurons.get('active', 0):,} neurons are active (sparse coding).")
                    print(f"  My mood is {mood}.")
                    print(f"  I have processed {brain.state.simulation_step} interactions.")
                    
                    if personality:
                        top_trait = max(personality.items(), key=lambda x: x[1])
                        print(f"  My strongest personality trait is {top_trait[0]} ({top_trait[1]:.2f}).")
                    print()

                elif cmd == '/chemicals':
                    data = brain.get_dashboard_data()
                    print(f"\n{Colors.BOLD}Neuromodulator State:{Colors.RESET}")
                    for chem, level in data.get('chemicals', {}).items():
                        color = colorize_level(level)
                        bar = make_bar(level)
                        print(f"  {chem:15} {color}[{bar}]{Colors.RESET} {level:.2f}")
                    print(f"\n  Mood: {Colors.YELLOW}{data.get('mood', 'neutral')}{Colors.RESET}\n")

                elif cmd == '/stats':
                    stats = brain.get_stats()
                    print(f"\n{Colors.BOLD}Brain Statistics:{Colors.RESET}")
                    config = stats.get('config', {})
                    print(f"  Total neurons:     {config.get('total_neurons', 0):,}")
                    print(f"  Target sparsity:   {config.get('target_sparsity', 0.02):.2%}")
                    print(f"  Reservoir size:    {config.get('reservoir_size', 0):,}")
                    print(f"  Vocabulary size:   {config.get('vocabulary_size', 0):,}")
                    print(f"  Simulation steps:  {brain.state.simulation_step}")
                    print()

                elif cmd == '/regions':
                    data = brain.get_dashboard_data()
                    print(f"\n{Colors.BOLD}Brain Regions:{Colors.RESET}")
                    for region, info in data.get('regions', {}).items():
                        activity = info.get('activity', 0)
                        active = info.get('active_neurons', 0)
                        bar = make_bar(min(activity * 10, 1.0), width=15)
                        print(f"  {region:15} [{bar}] {active:4} active neurons")
                    print()

                elif cmd == '/save':
                    filepath = f"brain_state_{int(time.time())}.brain"
                    try:
                        brain.save(filepath)
                        print(f"{Colors.GREEN}Brain saved to {filepath}{Colors.RESET}\n")
                    except Exception as e:
                        print(f"{Colors.RED}Save failed: {e}{Colors.RESET}\n")

                elif cmd == '/clear':
                    clear_screen()

                elif cmd == '/help':
                    print_help()

                else:
                    print(f"{Colors.RED}Unknown command. Type /help for available commands.{Colors.RESET}\n")

            else:
                # Process through the brain
                result = brain.process(user_input)
                print_response(result['response'], result)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Interrupted. Use /quit to exit properly.{Colors.RESET}\n")

        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}\n")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
