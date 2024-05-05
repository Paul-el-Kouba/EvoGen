import os
import mido


def create_new_theme(generated_mid):
    template_mid = "./template/template.mid"

    """
        Copy Important Track Messages from the Generated Midi to the Theme Template
    """
    mid = mido.MidiFile(template_mid)
    mid2 = mido.MidiFile(generated_mid)

    mid.ticks_per_beat = mid2.ticks_per_beat

    for track in mid2.tracks:
        for msg in track:
            if hasattr(msg, 'channel'):
                if track.name == 'Guitar':
                    msg.channel = 0
                elif track.name == 'Strings':
                    msg.channel = 1

    mid.tracks[1] = [msg for msg in mid.tracks[1] if (msg.type != 'note_on' and msg.type != "end_of_track")]
    mid.tracks[2] = [msg for msg in mid.tracks[2] if (msg.type != 'note_on' and msg.type != "end_of_track")]

    mid2.tracks[3] = [msg for msg in mid2.tracks[3] if msg.type == 'note_on' or msg.type == 'end_of_track']
    mid2.tracks[4] = [msg for msg in mid2.tracks[4] if msg.type == 'note_on' or msg.type == 'end_of_track']

    for msg in mid2.tracks[3]:
        mid.tracks[1].append(msg.copy())
    for msg in mid2.tracks[4]:
        mid.tracks[2].append(msg.copy())

    if mid.tracks[1]:
        last_time_track_1 = max((msg.time for msg in mid.tracks[1] if msg.type == 'note_on'), default=0)
        while mid.tracks[1][-1].time != last_time_track_1:
            mid.tracks[1].pop()

    if mid.tracks[2]:
        last_time_track_1 = max((msg.time for msg in mid.tracks[2] if msg.type == 'note_on'), default=0)
        while mid.tracks[2][-1].time != last_time_track_1:
            mid.tracks[2].pop()

    """
        Clipping any bar >2; This is to keep the first two bars for the theme.
    """
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_bar = ticks_per_beat * 4
    first_two_bars_ticks = ticks_per_bar * 2

    for track in mid.tracks:
        new_track = []
        accumulated_ticks = 0
        for msg in track:
            if accumulated_ticks >= first_two_bars_ticks:
                break
            if accumulated_ticks + msg.time > first_two_bars_ticks:
                msg.time = first_two_bars_ticks - accumulated_ticks
            accumulated_ticks += msg.time
            new_track.append(msg)
        track.clear()
        track.extend(new_track)

    """
        Add note_off to any open note_on to remove errors    
    """
    for track in mid.tracks:
        active_notes = {}
        new_track = []
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[(msg.note, msg.channel)] = msg
            elif msg.type == 'note_on' and msg.velocity == 0:
                active_notes.pop((msg.note, msg.channel), None)
            elif msg.type == 'note_off':
                active_notes.pop((msg.note, msg.channel), None)
            new_track.append(msg)

        for note_msg in active_notes.values():
            note_off_msg = mido.Message('note_on', note=note_msg.note, velocity=0, channel=note_msg.channel,
                                        time=note_msg.time)
            new_track.append(note_off_msg)

        track.clear()
        track.extend(new_track)

    """
        Save the new theme to be evaluated
    """
    file_name = f"{generated_mid[:17]}/120_{generated_mid[18:]}"
    mid.save(file_name)
    os.remove(generated_mid)
    return file_name
