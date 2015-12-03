function [NNSETX,NNSETY]=nndetector_setup_inputs(NNSETX,NNSETY,SHOTGUN_MAX_SEC,SHOTGUN_SIGMA,TIMESTEP,SPECTROGRAMS,...
    FREQ_RANGE_DS,TIME_WINDOW_STEPS,NWINDOWS_PER_SONG,TARGET_OFFSETS,RANDOMSONGS,NMATCHINGSONGS,NTARGETS,TSTEP_OF_INTEREST)

nsongs=size(SPECTROGRAMS,1);
ntimes=size(SPECTROGRAMS,3);

shotgun = normpdf(0:TIMESTEP:SHOTGUN_MAX_SEC, 0, SHOTGUN_SIGMA);
shotgun = shotgun / max(shotgun);
shotgun = shotgun(find(shotgun>0.1));
shothalf = length(shotgun);

if shothalf
  shotgun = [ shotgun(end:-1:2) shotgun ];
end

% Populate the training data.  Infinite RAM makes this so much easier!

for song = 1:nsongs
  for tstep = TIME_WINDOW_STEPS : ntimes

    tmp=SPECTROGRAMS(RANDOMSONGS(song),...
    FREQ_RANGE_DS,...
    tstep-TIME_WINDOW_STEPS+1:tstep);

    NNSETX(:, (song-1)*NWINDOWS_PER_SONG + tstep - TIME_WINDOW_STEPS + 1) ...
      = reshape(tmp,[], 1);

    % Fill in the positive hits, if appropriate...

    if RANDOMSONGS(song) > NMATCHINGSONGS
      continue;
    end

    for interesting = 1:NTARGETS
      if tstep == TSTEP_OF_INTEREST(interesting)
        NNSETY(interesting, (song-1)*NWINDOWS_PER_SONG + tstep + TARGET_OFFSETS(interesting, RANDOMSONGS(song)) - TIME_WINDOW_STEPS - shothalf + 2 : ...
          (song-1)*NWINDOWS_PER_SONG + tstep + TARGET_OFFSETS(interesting, RANDOMSONGS(song)) - TIME_WINDOW_STEPS + shothalf) = shotgun;
      end
    end
  end
end
