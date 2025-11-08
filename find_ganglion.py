from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.board_controller import BoardController

def find_ganglion():
    print("Scanning for Ganglion boards (BLE)...")
    devices = BoardController.search_devices(BoardIds.GANGLION_BOARD.value)
    print(devices)

if __name__ == "__main__":
    find_ganglion()
