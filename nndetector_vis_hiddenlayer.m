function nndetector_vis_hiddenlayer(NET,FFT_TIME_SHIFT,TIME_WIN_STEPS,FREQ_RANGE,FREQ_RANGE_DS)
%
%
%

if NET.numLayers > 1
        for i = 1:size(NET.IW{1}, 1)
                subplot(size(NET.IW{1}, 1), 1, i)
                imagesc([-TIME_WIN_STEPS:0]*FFT_TIME_SHIFT*1000, linspace(FREQ_RANGE(1), FREQ_RANGE(2), length(FREQ_RANGE_DS))/1000, ...
                        reshape(NET.IW{1}(i,:), length(FREQ_RANGE_DS), TIME_WIN_STEPS));
                axis xy;
                ylabel('frequency');

                if i == 1
                        title('Hidden layers');
                end
                if i == size(NET.IW{1}, 1)
                        xlabel('time (ms)');
                end
                %imagesc(reshape(NET.IW{1}(i,:), TIME_WIN_STEPS, length(FREQ_RANGE_DS)));
        end
end
