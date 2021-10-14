import pretty_midi


# 0-87 note_on
# 88-175 note_off
# 176-275 time_shift

NOTE_ON_SIGN = 0
NOTE_OFF_SIGN = 88
TIME_SHIFT_SIGN = 176

class Event:
    def __init__(self,type,value):
        self.type = type
        self.value = value

    def __repr__(self):
        return 'type:{}--value:{}'.format(self.type, self.value)

    def get_value(self):
        return self.value

class Note:
    def __init__(self,time,type,pitch,velocity):
        self.time = time
        self.type = type
        self.pitch = pitch
        self.velocity = velocity

    def __repr__(self):
        return 'time:{}--type:{}--pitch:{}--velocity:{}'.format(self.time,self.type,self.pitch,self.velocity)

def create_time_shift_event(begin,end):
    time_shit_range = 100
    duration = int(round((end-begin)*100))

    num = duration // time_shit_range
    left = duration - num * time_shit_range
    events = []
    for i in range(num):
        events.append(Event('Time_shift',TIME_SHIFT_SIGN+time_shit_range-1))
    if left != 0:
        events.append(Event('time_shift',TIME_SHIFT_SIGN + left))

    return events


def Encode(filepath = None):
    midi= pretty_midi.PrettyMIDI(filepath)
    # only use the first instrument
    inst = midi.instruments[0]
    notes = inst.notes

    notes.sort(key=lambda x:x.start)

    my_notes = []
    for i in notes:
        my_notes.append(Note(i.start,'note_on',i.pitch,i.velocity))
        my_notes.append(Note(i.end, 'note_off', i.pitch, i.velocity))

    my_notes.sort(key=lambda x:x.time)

    my_events = []

    time_l = 0
    for i in my_notes:
        my_events += create_time_shift_event(time_l,i.time)
        time_l = i.time
        if i.type == 'note_on':
            my_events += [Event(i.type, i.pitch - 21)]
        elif i.type == 'note_off':
            my_events += [Event(i.type, NOTE_OFF_SIGN + i.pitch - 21)]

    array = [i.get_value() for i in my_events]

    return array

def Decode(array,file_path,save=False):
    time = 0
    notes_dic = {}
    all_notes = []
    for i in array:
        if i >=176:
            time += (i-176)/100
            continue
        elif i < 88:
            notes_dic[i+21] = time
        else:
            try:
                t = notes_dic[i-88+21]
                if t - time == 0:
                    continue
                note = pretty_midi.Note(60, i-88+21, t, time)
                all_notes.append(note)

            except:
                print('pitch does not match : {}'.format(i-88+21))
    all_notes.sort(key=lambda x: x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By Carl")
    instument.notes = all_notes

    mid.instruments.append(instument)

    if file_path is not None and save:
        mid.write(file_path)

    return mid



# Test code
# a = Encode('dataset/data/train/000.midi')
#
# Decode(a)