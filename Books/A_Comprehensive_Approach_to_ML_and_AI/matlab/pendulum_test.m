% Create an instance of the environment.
env = InvertedPendulum();

% Reset the environment.
obs = env.reset();
done = false;

while ~done
    % Choose a random action between -maxTorque and maxTorque.
    action = -env.maxTorque + 2 * env.maxTorque * rand;
    
    % Take a step.
    [obs, reward, done] = env.step(action);
    
    % Display the state and reward.
    disp(['Observation: ', mat2str(obs, 3), ', Reward: ', num2str(reward)]);
    
    % Optionally render the pendulum.
    env.render();
    
    % Pause for visualization (optional).
    % pause(0.001);
end

disp('Episode finished.');
