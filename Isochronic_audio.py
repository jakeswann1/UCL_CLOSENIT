import numpy as np
import pygame
import sounddevice
import random
import json


###The yputube videos had a range from 126 to 135 for alpha and 100 and 129 for beta
###callback function, that calculates the start and stop angle of each wave to make a continouos wave to remove harmonics

#Have slowed down the delay from Lan's opinion
class sham:
    def __init__(self, 
                # parent_current, parent_desired, Cur_Queue, Des_Queue
                 ):
        self.freq = 200
        self.mod_freq = 10
        self.sample_rate = 48_000
        self.increment = 0.1
        self.phase = 0
        self.phaseC = 0
        self.dummy_BB = 0
        self.dummy_DATA = 0
        self.thresh = 0
        self.phase_mod = 0
        self.mod_depth = 1


        # self.parent_current = parent_current
        # self.parent_desired = parent_desired
        # self.Cur_Queue = Cur_Queue
        # self.Des_Queue = Des_Queue

        pygame.init()
        pygame.display.set_mode(size=(320, 240))
        pygame.display.set_caption('Playing Musics')

    def audio_callback(self, 
        outdata: np.ndarray, frames: int, time: 'CData', status: sounddevice.CallbackFlags,
    ) -> None:
        omega = 2*np.pi*self.freq/self.sample_rate #Angular frequency
        start_angle = self.phase
        stop_angle = self.phase + frames*omega
        #constant_angle = phase + frames*2*np.pi*cons_freq/sample_rate
        self.phase = np.fmod(stop_angle, 2*np.pi)

        arg = np.linspace(
            start=start_angle,
            stop=stop_angle,
            num=frames,
        ) #Linear interpolation between the start and stop angle which can be input into the sine wave below
        # omegaC = 2*np.pi*self.cons_freq/self.sample_rate #Angular frequency
        # start_angleC = self.phaseC
        # stop_angleC = self.phaseC + frames*omegaC
        # self.phaseC = np.fmod(stop_angleC, 2*np.pi)

        # argC = np.linspace(
        #     start=start_angleC,
        #     stop=stop_angleC,
        #     num=frames,
        # ) 

        # Modulation frequency (creates the pulsing effect)
        omega_mod = 2*np.pi*self.mod_freq/self.sample_rate
        start_angle_mod = self.phase_mod
        stop_angle_mod = self.phase_mod + frames*omega_mod
        self.phase_mod = np.fmod(stop_angle_mod, 2*np.pi)

        arg_mod = np.linspace(
            start=start_angle_mod,
            stop=stop_angle_mod,
            num=frames,
        )
        
        # Create the carrier signal
        carrier = np.sin(arg)
        
        # Create the modulation signal (oscillates between 0 and 1)
        modulation = 0.5 * (1 + self.mod_depth * np.sin(arg_mod))
        
        # Apply modulation to create isochronic tone
        isochronic_signal = carrier * modulation
        
        # Apply to both channels with appropriate amplitude
        outdata[:, 0] = 10_000 * isochronic_signal
        outdata[:, 1] = 10_000 * isochronic_signal

    def set_modulation_frequency(self, freq: float) -> None:
        """Set the modulation frequency directly"""
        self.mod_freq = max(1, min(self.sample_rate//2, freq))
        print(f'Modulation Frequency set to: {self.mod_freq:.1f} Hz')
    
    def set_modulation_depth(self, depth: float) -> None:
        """Set the modulation depth (0-1)"""
        self.mod_depth = max(0, min(1, depth))
        print(f'Modulation depth set to: {self.mod_depth:.2f}')
    
    def set_carrier_frequency(self, freq: float) -> None:
        """Set the carrier frequency"""
        self.freq = max(1, min(self.sample_rate//2, freq))
        print(f'Carrier frequency set to: {self.freq:.1f} Hz')


    def main(self) -> None:
        #Initiate pygame window

        with sounddevice.OutputStream(
            callback=self.audio_callback, channels=2, samplerate=self.sample_rate, dtype='int16',
        ) as stream:
            stream.start()
            #self.step_init()
            while True:

                # if np.round(pygame.time.get_ticks()/1000, 1)- self.thresh > 5:
                #     self.recorder()
                #     self.thresh = np.round(pygame.time.get_ticks()/1000, 1)

                
                pygame.time.delay(200)
                # Here James code will adjust the value of the modulation
                # x = random.choice([-1, 1])
                
                # if x == -1:
                #     if self.mod_freq <=8:
                #         self.set_modulation_frequency(1)
                #     else:
                #         self.set_modulation_frequency(-1)
                # else:
                #     if self.mod_freq >=14:
                #         self.set_modulation_frequency(-1)
                #     else:
                #         self.set_modulation_frequency(1)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_DOWN]:
                    self.set_modulation_frequency(self.mod_freq-1)
                elif pressed[pygame.K_UP]:
                    self.set_modulation_frequency(self.mod_freq+1)


if __name__ == '__main__':
    sham_BB = sham()
    sham_BB.main()
