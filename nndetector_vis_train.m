function AX=nndetector_vis_train(TIMES,FREQS,AVE_SPECT,TARGETS,TARGETS_STEPS,FREQ_RANGE,TIME_WIN)
%
%
%% Draw the pretty full-res spectrogram and the targets
% figure(4);
% subplot(2,1,1);
% imagesc([TIMES(1) TIMES(end)], [FREQS(1) FREQS(end)]/1000, AVE_SPECT);
% axis xy;
% xlabel('Time (ms)');
% ylabel('Frequency (kHz)');
% colorbar;

%% Draw the pretty full-res spectrogram and the targets
AX=subplot(length(TARGETS_STEPS)+1,1,1);
imagesc([TIMES(1) TIMES(end)], [FREQS(1) FREQS(end)]/1000, AVE_SPECT);
axis xy;
xlabel('Time (ms)');
ylabel('Frequency (kHz)');

% Draw the syllables of interest:
line(repmat(TARGETS, 2, 1), repmat([FREQS(1) FREQS(end)]/1000, TARGETS_STEPS, 1)', 'Color', [1 0 0]);

windowrect = rectangle('Position', [(TARGETS(1) - TIME_WIN) ...
                                    FREQ_RANGE(1)/1000 ...
                                    TIME_WIN(1) ...
                                    (FREQ_RANGE(2)-FREQ_RANGE(1))/1000], ...
                       'EdgeColor', [1 0 0]);

drawnow;
