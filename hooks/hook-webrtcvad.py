from PyInstaller.utils.hooks import copy_metadata

# webrtcvad-wheels installs the package metadata under 'webrtcvad-wheels',
# not 'webrtcvad', so the standard hook's copy_metadata('webrtcvad') fails.
datas = copy_metadata('webrtcvad-wheels')
