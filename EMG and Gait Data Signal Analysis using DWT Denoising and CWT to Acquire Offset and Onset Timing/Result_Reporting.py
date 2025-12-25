def generate_cycle_report(segment):
    """
    Generates a detailed text report for a single gait cycle
    Calculates Onset/Offset in % Gait Cycle and Duration in ms
    """
    report = []
    cycle_id = segment['cycle_id']
    fs = segment['fs']
    
    # Absolute time boundaries of the cycle
    t_start = segment['time'][0]
    t_end = segment['time'][-1]
    cycle_duration = t_end - t_start
    
    report.append(f"========================================")
    report.append(f" GAIT CYCLE {cycle_id}")
    report.append(f"========================================")
    report.append(f"Total Duration : {cycle_duration:.3f} s")
    report.append(f"Sampling Rate  : {int(fs)} Hz")
    report.append("")
    
    # Helper function to process muscle data
    def process_muscle(muscle_name, activation_list):
        report.append(f"[{muscle_name}]")
        if not activation_list:
            report.append("  No activation detected.")
            return

        for i, act in enumerate(activation_list):
            # Get absolute times
            abs_onset = act['start_t']
            abs_offset = act['end_t']
            
            # Convert to % Gait Cycle
            # Formula: (t - t_start) / duration * 100
            pct_onset = ((abs_onset - t_start) / cycle_duration) * 100
            pct_offset = ((abs_offset - t_start) / cycle_duration) * 100
            
            # Calculate Duration (ms)
            dur_ms = (abs_offset - abs_onset) * 1000
            
            # Format string
            report.append(f"  > Burst {i+1}:")
            report.append(f"    Onset  : {pct_onset:6.2f} % GC")
            report.append(f"    Offset : {pct_offset:6.2f} % GC")
            report.append(f"    Duration: {dur_ms:6.2f} ms")
        report.append("")

    # Process GL
    activations_gl = segment.get('activations_gl', [])
    process_muscle("Gastrocnemius Lateralis (GL)", activations_gl)
    
    # Process VL
    activations_vl = segment.get('activations_vl', [])
    process_muscle("Vastus Lateralis (VL)", activations_vl)
    
    return "\n".join(report)

def generate_full_summary(segments):
    """
    Generates a summary report for all cycles
    Useful for saving to file (future feature)
    """
    full_report = ""
    for seg in segments:
        full_report += generate_cycle_report(seg)
        full_report += "\n\n"
    return full_report