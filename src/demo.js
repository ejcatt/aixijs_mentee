const demo = {
	ui: new UI(),

	new(config) {
		if (this.vis) this.vis.pause();

		// get defaults
		for (let opt of ['env', 'agent']) {
			let lst = [];
			for (let ptr = config[opt].type; ptr; ptr = ptr.__proto__) {
				if (!ptr.params) {
					continue;
				}

				for (let p of ptr.params) {
					lst.push(p);
				}
			}

			for (let i = lst.length - 1; i >= 0; i--) {
				let v = config[opt][lst[i].field];
				if (!v) {
					config[opt][lst[i].field] = lst[i].value;
				}
			}
		}

		this.ui.hide('picker');
		this.ui.show('setup');

		this.ui.clear();
		this.ui.push(config);

		let label = this.ui.getElementById('setup_label');
		label.innerText = `Setup: ${config.name}`;

		this.config = config;

		this.ui.clearExplanations();
		if (!config.vis.exps) return;
		for (let exp of config.vis.exps) {
			this.ui.showExplanation(exp);
		}

		if (!config.exps) return;

		for (let exp of config.exps) {
			this.ui.showExplanation(exp);
		}
	},

	run(novis, env) {
		this.novis = novis;

		// new: class defaults -> config -> ui
		// run: ui -> options -> env/agent
		if (this.vis) this.vis.remove();
		let options = Util.deepCopy(this.config);
		this.nolivevis = options.nolivevis;
		this.ui.pull(options);
		this.env = env ? env : new options.env.type(options.env);

		if (options.env._mods) {
			options.env._mods(this.env);
		}

		options.agent.model = this.env.makeModel(DirichletGrid, options.agent.modelParametrization);
		options.agent.numActions = this.env.numActions || this.env.actions.length;
		options.agent.min_reward = this.env.min_reward;
		options.agent.max_reward = this.env.max_reward;

		options.agent.discountParam = options.agent.discountParam || { gamma: 0.99 };

		this.agent = new options.agent.type(options.agent);
		if (options.agent._mods) {
			options.agent._mods(this.agent);
		}

		this.trace = new this.agent.tracer(options.agent.steps);

		this.plots = [];
		Plot.clearAll();
		for (let P of this.trace.plots) {
			this.plots.push(new P(this.trace));
		}

		for (let P of this.env.plots) {
			this.plots.push(new P(this.trace));
		}

		let callback = _ => {
			this.ui.end();
			this.cancel = false;
			let frames = this.trace.iter;
			let second = Util.roundTo((performance.now() - this.t0) / 1000, 2);
			let fps = Util.roundTo(frames / second, 2);
			this.trace.runtime = second;
			this.trace.fps = fps;
			console.log(`${frames} steps, ${second} seconds (${fps} fps)`);

			if (!novis && options.vis) {
				this.vis = new options.vis(this.env, this.trace, this.ui);
				this.vis.reset();
			}

			for (let p of this.plots) {
				p.dataGUIUpdate();
			}
		};
		if (!novis) {
			this.ui.start();
		}

		this.t0 = performance.now();
		this._simulate(callback);
	},

	_simulate(callback) {
		let trace = this.trace;
		let agent = this.agent;
		let env = this.env;

		let e = env.generatePercept();
		let a = env.noop;

		trace.log(agent, env, a, e);
		agent.update(a, e);

		let step = _ => {
			trace.log(agent, env, a, e);
			a = agent.selectAction(e);
			env.perform(a);
			e = env.generatePercept();
			agent.update(a, e);
		};

		if (this.nolivevis || this.novis) {
			while (true) {
				if (trace.iter == trace.t || this.cancel) {
					callback();
					return;
				}
				step();
				for (let p of this.plots) {
					p.dataUpdate(trace);
				}
			}
		}

		loop = _ => {
			if (trace.iter == trace.t || this.cancel) {
				callback();
				return;
			}

			step();
			for (let p of this.plots) {
				p.dataUpdate(trace);
				p.dataGUIUpdate();
			}
			setTimeout(loop, 0);
		};

		loop();
	},

	stop() {
		this.cancel = true;
	},

	reset() {
		if (this.vis) this.vis.remove();
		this.ui.end();
		this.ui.hide('navigation');
		this.ui.hide('plots');
		this.ui.show('picker');
		this.ui.hide('setup');
		this.ui.clearExplanations();
	},

	experiment(dems, params) {
		this.reset();
		this.experiment_number ? this.experiment_number++ : this.experiment_number = 1;
		if (!params) {
			// some defaults
			params = {
				runs: 1,
				env: { N: 10 },
				agent: { steps: 200 },
				download: false,
			};
		}
		var runs = params.runs || 1;
		var frac = params.frac || 1;
		results = {};
		seed = params.seed || 'aixi';
		let num = 1;
		var t0 = performance.now();
		for (let cfg of dems) {
			let config = Util.deepCopy(cfg);
			if (params.env) {
				for (let param_name in params.env) {
					config.env[param_name] = params.env[param_name];
				}
			}
			if (params.agent) {
				for (let param_name in params.agent) {
					config.agent[param_name] = params.agent[param_name];
				}
			}
			if (config.agent.model) {
				console.log(`Running ${config.agent.type.name} with model ${config.agent.model.name} on ${config.env.type.name}.`);
			} else {
				console.log(`Running ${config.agent.type.name} on ${config.env.type.name}.`);
			}

			Math.seedrandom(seed);
			logs = [];
			let env = null;
			this.new(config);
			for (let i = 0; i < runs; i++) {
				console.log(`    run ${i + 1} of ${runs}...`);
				if (i > 0) {
					env = new env.constructor(env.options);
					if (env.constructor == Gridworld) {
						env.isSolvable();
					}

					this.run(true, env);
				} else {
					this.run(true);
					env = this.env;
				}

				var rew = [];
				var exp = [];
				for (var j = 0; j < config.agent.steps; j++) {
					if (j % frac == 0) {
						rew.push(this.trace.averageReward[j]);
						exp.push(this.trace.explored[j]);
					}
				}

				var log = {
					rewards: rew,
					explored: exp,
					options: Util.deepCopy(this.config),
					agent: this.config.agent.type.name,
					steps: this.trace.iter,
					runtime: this.trace.runtime,
					samples: this.agent.samples,
					horizon: this.agent.horizon,
					seed: seed,
					gamma: this.agent.gamma,
					epsilon: this.agent.epsilon,
				};

				// TODO: refactor logging to decouple this
				if (this.agent.tracer == RewardCorruptionTrace) {
					var crew = [];
					var trew = [];
					for (let j = 0; j < config.agent.steps; j++) {
						if (j % frac == 0) {
							crew.push(this.trace.averageCorruptReward[j]);
							trew.push(this.trace.averageTrueReward[j]);
						}
					}
					log.corrupt_rewards = crew;
					log.true_rewards = trew;
				}

				logs.push(log);
			}
			var key = '';
			if (config.name in results) {
				key = `${config.name}-${num}`;
				num++;
			} else {
				key = config.name;
			}
			results[key] = logs;
		}
		this.reset();

		console.log(`Done! Total time elapsed: ${Math.floor(performance.now() - t0) / 1000} seconds.`);

		let json = JSON.stringify(results);
		let blob = new Blob([json], { type: 'application/json' });

		let a = document.createElement('a');
		a.download = `results-${this.experiment_number}.json`;
		a.href = URL.createObjectURL(blob);
		a.textContent = `Download results-${this.experiment_number}.json`;
		if (params.download) {
			a.dispatchEvent(new MouseEvent('click'));
		}
		document.body.appendChild(a);

		return results;
	},
};
