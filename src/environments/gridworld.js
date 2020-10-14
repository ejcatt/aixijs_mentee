class Gridworld extends Environment {
	constructor(options) {
		super(options);
		if (!options.randomized) {
			let Cl = this.constructor;
			options.randomized = true;
			return Gridworld.generateRandom(Cl, options);
		}

		this.rewards = options.rewards || Gridworld.rewards;

		this.plots = [ExplorationPlot];
		this.obsBits = 4;
		this.grid = [];
		this.N = options.N;
		Util.assert(this.N);
		this.state = {};
		this.actions = Gridworld.actions;
		this.numActions = this.actions.length;
		this.reward = 0; // fix name conflict
		this.noop = 4;
		this.visits = 0;
		this.state_percepts = options.state_percepts;
		this.initialQ = options.initialQ || 100;


		this.min_reward = Math.min(this.rewards.wall,this.rewards.trap) + this.rewards.move;
		this.max_reward = this.rewards.chocolate + this.rewards.move;

		for (let i = 0; i < this.N; i++) {
			this.grid[i] = new Array(this.N);
			for (let j = 0; j < this.N; j++) {
				this.grid[i][j] = Gridworld.newTile(i, j, options, options.map[j][i], this.rewards);
			}
		}

		if (options.goals) {
			this.goals = [];
			for (let goal of options.goals) {
					let type = goal.type || Gridworld.map_symbols.dispenser;
					let g = Gridworld.newTile(goal.x, goal.y, goal.freq, type, this.rewards);
					g.goal = true;
					this.grid[goal.x][goal.y] = g;
					this.goals.push(g);
			}
		}

		this.generateConnexions();

		if (options.initial) {
			this.pos = this.grid[options.initial.x][options.initial.y];
		} else {
			this.pos = this.grid[0][0];
		}
		
		
		this.traps = [ (!this.pos.connexions[0]) ? 0 : this.pos.connexions[0].constructor, (!this.pos.connexions[1]) ? 0 : this.pos.connexions[1].constructor,  (!this.pos.connexions[2]) ? 0 : this.pos.connexions[2].constructor ,  (!this.pos.connexions[3]) ? 0 : this.pos.connexions[3].constructor]; 
		this.grid_locations = new Array();

	}

	generateConnexions() {
		let grid = this.grid;
		let actions = this.actions;
		grid.forEach((row, idx) => {
			row.forEach((tile, jdx) => {
				let str = '';
				for (let a = 0; a < this.numActions; a++) {
					let i = actions[a][0];
					let j = actions[a][1];
					if (!grid[idx + i] ||
						!grid[idx + i][jdx + j] ||
						grid[idx + i][jdx + j].constructor == Wall) {
						str += '1';
					} else {
						if (i || j) {
							str += '0';
						}

						if (tile.constructor != Trap && tile.constructor != Wall) {
							tile.connexions[a] = grid[idx + i][jdx + j];
						}

					}
				}
				if (this.state_percepts ) {
					tile.obs = idx * this.N + jdx;
				} else {
					tile.obs = parseInt(str, 2);
				}

			});
		});
	}

	isSolvable() {
		let queue = [];
		let pos = 0;

		let maxFreq = 0;
		for (let goal of this.options.goals) {
			if (goal.freq > maxFreq) {
				maxFreq = goal.freq;
			}
		}

		for (let i = 0; i < this.N; i++) {
			for (let j = 0; j < this.N; j++) {
				this.grid[i][j].expanded = false;
			}
		}

		this.numStates = 1;
		queue.push(this.grid[0][0]);
		let solvable = false;
		while (pos < queue.length) {
			let ptr = queue[pos];
			ptr.expanded = true;
			for (let t of ptr.connexions) {
				if (!t || t.expanded) {
					continue;
				}

				this.numStates++;
				if ((t.constructor == Dispenser && t.freq == maxFreq) || t.constructor == Chocolate) {
					solvable = true;
				}

				t.expanded = true;
				queue.push(t);
			}

			pos++;
		}

		return solvable;
	}

	static generateRandom(Cl, options) {
		let opt = Gridworld.proposeRandom(options);
		let env = new Cl(opt);
		if (!env.isSolvable()) {
			return Gridworld.generateRandom(Cl, options);
		}

		return env;

	}

	static proposeRandom(options) {
		let opt = Util.deepCopy(options);
		let N = options.N;
		let trapProb = options.trapProb || 0.2;
		let wallProb = options.wallProb || 0.25;
		opt.map = [];
		if (options._set_seed) {
			Math.seedrandom('foo');
		}
		for (let i = 0; i < N; i++) {
			opt.map[i] = new Array(N);
			for (let j = 0; j < N; j++) {
				if (i == 0 && j == 0) {
					opt.map[i][j] = Gridworld.map_symbols.empty;
				}

				let r = Math.random();
				if (r < trapProb) {
					opt.map[i][j] = Gridworld.map_symbols.trap;
				} else if (r < wallProb) {
					opt.map[i][j] = Gridworld.map_symbols.wall;
				} else {
					opt.map[i][j] = Gridworld.map_symbols.empty;
				}
			}
		}

		// Adding prob 1/5 goals
		for (let i = 0; i < 2*N; i++){
			for (let goal of opt.goals) {
					let g = Gridworld.proposeGoal(N);
					goal.x = g.x;
					goal.y = g.y;
					opt.map[g.y][g.x] = Gridworld.map_symbols.chocolate;
			}
		}

		return opt;
	}

	static proposeGoal(N) {
		let gx = Util.randi(0, N);
		let gy = Util.randi(0, N);
		if (gx + gy < N / 2) {
			return Gridworld.proposeGoal(N);
		}

		return {
			x: gx,
			y: gy,
		};

	}

	static newTile(i, j, info, type, rewards) {
		let tile;
		if (type == Gridworld.map_symbols.empty) {
			tile = new Tile(i, j, rewards.empty);
		} else if (type == Gridworld.map_symbols.wall) {
			tile = new Wall(i, j, rewards.wall);
		} else if (type == Gridworld.map_symbols.chocolate) {
			tile = new Chocolate(i, j, rewards.chocolate);
		} else if (type == Gridworld.map_symbols.dispenser) {
			tile = new Dispenser(i, j, info, rewards.chocolate, rewards.empty);
		} else if (type == Gridworld.map_symbols.trap) {
			tile = new Trap(i, j, rewards.trap);
		} else if (type == Gridworld.map_symbols.modifier) {
			tile = new SelfModificationTile(i, j, rewards.modifier);
		} else {
			throw `Error: unknown Tile type: ${type}.`;
		}

		return tile;
	}

	perform(action) {
		var rew = this.rewards.move;
		var t = this.pos.connexions[action];

		if (t) {
			rew += t.reward();
			if (!t.visited) {
				t.visited = true;
				this.visits++;
			}

			this.pos = t;
			this.wall_hit = false;
		} else {
			if (this.pos.constructor == Trap) {
				rew += this.rewards.trap;
			} else {
				rew += this.rewards.wall;
				this.wall_hit = true;
			}
		}
		
		this.traps = [ (!this.pos.connexions[0]) ? 0 : this.pos.connexions[0].constructor, (!this.pos.connexions[1]) ? 0 : this.pos.connexions[1].constructor,  (!this.pos.connexions[2]) ? 0 : this.pos.connexions[2].constructor ,  (!this.pos.connexions[3]) ? 0 : this.pos.connexions[3].constructor]; 
		this.grid_locations = t;

		rew += this.dynamics(this.pos);
		this.reward = rew;
	}

	dynamics(tile) {
		tile.dynamics();
		return 0;
	}

	generatePercept() {
		return {
			obs: this.pos.obs,
			rew: this.reward,
			traps: this.traps
		};
	}

	save() {
		this.state = {
			x: this.pos.x,
			y: this.pos.y,
			reward: this.reward,
		};
	}

	load() {
		Util.assert(this.state, 'No saved state to load!');
		this.pos = this.grid[this.state.x][this.state.y];
		this.reward = this.state.reward;
	}

	copy() {
		let res = new this.constructor(this.options);
		res.pos = res.grid[this.pos.x][this.pos.y];
		res.reward = this.reward;

		return res;
	}

	getState() {
		return { x: this.pos.x, y: this.pos.y };
	}

	makeModel(model, parametrization) {
		if (model == QTable) {
			return new QTable(this.initialQ, this.numActions);
		}

		if (model == DirichletGrid) {
			return new DirichletGrid(this.options.N);
		}

		let modelClass = [];
		let modelWeights = [];
		let options = Util.deepCopy(this.options);

		// remove traps from agents model
		for (let k = 0; k < options.N; k++) {
			for (let l = 0; l < options.N; l++) {
				//options.grid[i][j] = new Tile(i, j, m.rewards.empty);
				if (options.map[k][l] == Gridworld.map_symbols.trap) {
					options.map[k][l] = Gridworld.map_symbols.empty;
				}
			}
		}

		if (parametrization == 'mu') {
			modelClass.push(new this.constructor(options));
			modelWeights = [1];
		} else if (parametrization == 'maze') {
			options.randomized = false;
			for (let n = 4; n < this.options.N; n++) {
				options.N = n;
				for (let k = 0; k < n; k++) {
					modelClass.push(Gridworld.generateRandom(this.constructor, options));
					modelWeights.push(1);
				}
			}

			modelClass.push(new this.constructor(this.options));
			modelWeights.push(1);
		} else {
			let C = options.N * options.N;
			modelWeights = Util.zeros(C);
			
			
			for (let i = 0; i < options.N; i++) {
				for (let j = 0; j < options.N; j++) {
					if (parametrization == 'goal') {
						options.goals = [
							{
								x: j,
								y: i,
								freq: options.goals[0].freq,
							},
						];
					} else if (parametrization == 'pos') {
						options.initial = { x: j, y: i };
					}

					let t = this.grid[j][i];
					if (t.constructor == Wall || !t.expanded) {
						modelWeights[i * options.N + j] = 0;
					} else {
						modelWeights[i * options.N + j] = 1 / C; // default uniform
					}

					let m = new this.constructor(options);

					modelClass.push(m);
				}
			}
		}

		// ensure prior is normalised
		let C = modelWeights.length;
		let s = Util.sum(modelWeights);
		for (let i = 0; i < C; i++) {
			modelWeights[i] /= s;
		}

		return new BayesMixture(modelClass, modelWeights);
	}

	conditionalDistribution(e) {
		let p = this.generatePercept();
		let s = this.pos;
		if (s.constructor == NoiseTile) {
			return e.rew == p.rew ? s.prob : 0;
		}

		if (e.obs != p.obs) {
			// observations are deterministic
			return 0;
		} else if (!s.goal) {
			// all tiles except the goal are deterministic
			return e.rew == p.rew ? 1 : 0;
		} else {
			let rew = e.rew - this.rewards.move;
			if (rew == this.rewards.chocolate) {
				return s.freq;
			} else if (rew == this.rewards.empty) {
				return 1 - s.freq;
			} else {
				return rew == this.rewards.wall && this.wall_hit;
			}
		}
	}
}

Gridworld.actions = [
	[-1, 0], // left
	[1, 0], // right
	[0, -1], // up
	[0, 1], // down
	[0, 0], // noop
];

Gridworld.rewards = {
	chocolate: 1,
	wall: 0,
	empty: 0,
	move: 0,
	trap: -100,
};

Gridworld.params = [
	{ field: 'N', value: 10 },
	{ field: 'goals', value: [{ freq: 0.75 },] },
];

Gridworld.map_symbols = {
	empty: 'F',
	chocolate: 'C',
	wall: 'W',
	dispenser: 'D',
	sign: 'S',
	trap: 'T',
	modifier: 'M',
};

class WireheadingGrid extends Gridworld {
	dynamics(tile) {
		if (tile.constructor == SelfModificationTile) {
			this.conditionalDistribution = e => {
				let p = this.generatePercept();
				return p.rew == e.rew;
			};

			this.generatePercept = _ => {
				let p = super.generatePercept();
				p.rew = Number.MAX_SAFE_INTEGER;
				return p;
			};

			this.wireheaded = true;
		}

		return 0;
	}

	getState() {
		let s = super.getState();
		s.wireheaded = this.wireheaded;

		return s;
	}

	save() {
		super.save();
		this.saved_conditionalDistribution = this.conditionalDistribution;
		this.saved_generatePercept = this.generatePercept;
	}

	load() {
		super.load();
		this.conditionalDistribution = this.saved_conditionalDistribution;
		this.generatePercept = this.saved_generatePercept;
	}
}

class EpisodicGrid extends Gridworld {
	conditionalDistribution(e) {
		let p = this.generatePercept();
		return (e.obs == p.obs && e.rew == p.rew) ? 1 : 0;
	}

	dynamics(tile) {
		if (tile.constructor == Chocolate) {
			this.pos = this.grid[0][0];
		}

		return 0;
	}
}

class Tile {
	constructor(x, y, r) {
		this.x = x;
		this.y = y;
		this.rew = r;
		this.reward = function () { return this.rew; };

		this.legal = true;
		this.color = GridVisualization.colors.empty;
		this.info = [];
		this.obs = null; // gets filled out on construction
		this.symbol = 0; // what it looks like from afar
		this.connexions = new Array();
		this.dynamics = _ => { };
	}
}

class Wall extends Tile {
	constructor(x, y, r) {
		super(x, y, r);

		this.legal = false;
		this.color = GridVisualization.colors.wall;
		this.symbol = 1;
	}
}

class Chocolate extends Tile {
	constructor(x, y, r) {
		super(x, y, r);
		this.color = GridVisualization.colors.chocolate;
	}
}

class Dispenser extends Tile {
	constructor(x, y, freq, r1, r2) {
		super(x, y, r1);
		this.rew2 = r2;
		this.freq = freq;
		this.color = GridVisualization.colors.dispenser;
		this.reward = function () {
			return Math.random() < this.freq ? this.rew : this.rew2;
		};
	}
}

class Trap extends Tile {
	constructor(x, y, r) {
		super(x, y, r);
		this.color = GridVisualization.colors.trap;
	}
}

class SelfModificationTile extends Tile {
	constructor(x, y, r) {
		super(x, y, r);
		this.color = GridVisualization.colors.modifier;
	}
}

class NoiseTile extends Tile {
	constructor(x, y, r) {
		super(x, y, r);
		this.numObs = Math.pow(2, 2);
		this.prob = 1 / this.numObs;
		this.color = GridVisualization.colors.noise;
		this.dynamics = function () {
			this.obs = Util.randi(0, this.numObs);
		};
	}
}
