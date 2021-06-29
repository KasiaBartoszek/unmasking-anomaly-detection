#!/usr/bin/env python
import sys
import PySimpleGUI as sg
from main import run

def MachineLearningGUI():
    sg.set_options(text_justification='right')

    folder_picker = [[
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ]]

    file_picker = [[
        sg.Text("Video File"),
        sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
        sg.FileBrowse(),
    ]]

    foldersection = [[sg.Frame('Pick folder', folder_picker, title_color='green', font='Any 12', key='folderpicker')]]
    filesection =[[sg.Frame('Pick file', file_picker, title_color='green', font='Any 12', key='filepicker')]]
    
    command_line_parms = [[sg.Text('k', size=(8, 1)), sg.Spin(values=[i for i in range(1, 1000)], initial_value=10, size=(6, 1), key='k' ),
                           sg.Text('m', size=(8, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 1000)], initial_value=4, size=(6, 1), key='m')]]

    layout = [[sg.Checkbox('Video instead of frames', default=False, enable_events=True, key='checkbox')],
            [collapse(foldersection, 'foldersection', True)],
            [collapse(filesection, 'filesection', False)],
            [sg.Frame('Command Line Parameteres', command_line_parms, title_color='green', font='Any 12')],
            [sg.Submit(), sg.Cancel()],
            [sg.Output(size=(110,30), background_color='black', text_color='white')]]

    sg.set_options(text_justification='left')

    window = sg.Window('Anomaly Detection',
                       layout, font=("Helvetica", 12))
    folder = ''
    path = 'path'
    while True:             # Event Loop
        event, values = window.Read()
        if event in (None, 'Cancel'):
            break
        if event == 'Submit':
            run(type('',(),{path:folder, 'k':values['k'],'m':values['m'] })())
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
        if event == "-FILE-":
            folder = values["-FILE-"]
        if values['checkbox'] == True:
            window['filesection'].Update(visible = True)
            window['foldersection'].Update(visible = False)
            path = 'video_path'
        if values['checkbox'] == False:
            window['filesection'].Update(visible = False)
            window['foldersection'].Update(visible = True)
            path = 'path'
    window.Close()

def collapse(layout, key, visible):
    return sg.pin(sg.Column(layout, key=key, visible=visible))

def CustomMeter():
    # layout the form
    layout = [[sg.Text('Loading')],
              [sg.ProgressBar(1000, orientation='h',
                              size=(20, 20), key='progress')],
              [sg.Cancel()]]

    # create the form`
    window = sg.Window('Anomaly Detection', layout)
    progress_bar = window['progress']
    # loop that would normally do something useful
    for i in range(1000):
        # check to see if the cancel button was clicked and exit loop if clicked
        event, values = window.read(timeout=0, timeout_key='timeout')
        if event == 'Cancel' or event == None:
            break
        # update bar with loop value +1 so that bar eventually reaches the maximum
        progress_bar.update_bar(i+1)
    # done with loop... need to destroy the window as it's still open
    window.CloseNonBlocking()


if __name__ == '__main__':
    CustomMeter()
    MachineLearningGUI()
