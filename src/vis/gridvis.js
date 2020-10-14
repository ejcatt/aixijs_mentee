class GridVisualization extends Visualization {
	constructor(env, trace, ui) {
		super(trace, ui);
		this.interval;
		this.time;
		this.grid = env.grid;
		this.N = env.N;
		this.rectangles = [];
		for (let i = 0; i < env.N; i++) {
			this.rectangles.push(new Array(env.N));
		}

		d3.selection.prototype.moveToFront = function () {
			return this.each(function () {
				this.parentNode.appendChild(this);
			});
		};

		var view_width = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
		var view_height = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);

		var dim = 0.5 * Math.min(view_width, view_height);
		var d = dim / this.N;
		if (!d) {
			d = GridVisualization.tile_size_px;
		}
		this.d = Math.min(d, GridVisualization.tile_size_px);
		this.width = (this.d + 1) * this.N - 1;
		this.height = (this.d + 1) * this.N - 1;
		this.svg
			.attr('width', this.width)
			.attr('height', this.height);

		this.grid.forEach((row, idx) => {
			row.forEach((tile, jdx) => {
				let r = GridVisualization.makeTile(this.svg, tile, undefined, this);
				this.rectangles[idx][jdx] = r;
			});
		});

		this.svg.append('image')
			.attr('xlink:href', 'assets/robot.svg')
			.attr('x', 0)
			.attr('y', 0)
			//.attr('width', this.d)
			.attr('height', this.d)
			.attr('id', 'agent');
	}

	updateEnv() {
		let x = this.pos_trace[this.time].x;
		let y = this.pos_trace[this.time].y;
		d3.select('#agent')
			.attr('x', (x + 0.2) * this.d)
			.attr('y', y * this.d);
	}

	static makeTile(svg, t, color, gv) {
		var d = GridVisualization.tile_size_px;
		if (gv) {
			d = gv.d;
		}
		let r = svg.append('rect')
			.attr('x', t.x * d)
			.attr('y', t.y * d)
			.attr('height', d)
			.attr('width', d)
			.attr('stroke', 'black')
			.attr('stroke-width', 2);
		if (t.constructor != Chocolate &&
			t.constructor != Dispenser) {
			r.attr('fill', t.color);
		} else {
			r.attr('fill', GridVisualization.colors.empty);
			GridVisualization.addCircle(svg, t.x, t.y, t.color, '', 0.75, gv);
		}

		if (color) {
			r.attr('fill', color);
		}

		return r;
	}

	static makeLegend(div, T, color) {
		let svg = d3.select(`#${div}`).append('svg')
			.attr('id', `${div}_svg`)
			.attr('width', GridVisualization.tile_size_px)
			.attr('height', GridVisualization.tile_size_px);
		GridVisualization.makeTile(svg, new T(0, 0), color);
	}

	static addCircle(svg, x, y, color, id, size, gv) {
		var d = GridVisualization.tile_size_px;
		if (gv) {
			d = gv.d;
		}
		svg.append('circle')
			.attr('cx', x * d + d / 2)
			.attr('cy', y * d + d / 2)
			.attr('r', size ? (d / 2) * size : d / 8)
			.attr('fill', color)
			.attr('stroke', '#000')
			.attr('id', id);
	}
}

GridVisualization.tile_size_px = 40;

GridVisualization.colors = {
	empty: '#fdfdfd',
	wall: 'grey',
	chocolate: 'orange',
	dispenser: 'orange',
	agent: 'blue',
	trap: 'pink',
	rho: 'red',
	modifier: 'blue',
};

class TabularGridVis extends GridVisualization {
	constructor(env, trace, ui) {
		super(env, trace, ui);
		this.arrows = [];
		for (let i = 0; i < env.N; i++) {
			this.arrows.push(new Array(env.N));
		}

		this.q_list = trace.q_map;
		this.arrow_actions = [0, 1, 2, 3];

		this.svg.append('defs').append('marker')
			.attr('id', 'arrowhead')
			.attr('refX', 3)
			.attr('refY', 2)
			.attr('markerWidth', 3)
			.attr('markerHeight', 4)
			.attr('orient', 'auto')
			.append('path')
			.attr('d', 'M 0,0 V 4 L2,2	 Z');

		this.grid.forEach((row, idx) => {
			row.forEach((tile, jdx) => {
				tile.info = [0, 0, 0, 0];
				if (!tile.legal || tile.goal) {
					this.arrows[idx][jdx] = null;
					return;
				}

				let xcoord = tile.x * this.d;
				let ycoord = tile.y * this.d;
				let arrowList = [];
				this.arrow_actions.forEach(a => {
					let arrow = this.svg.append('line')
						.attr('x1', xcoord)
						.attr('y1', ycoord)
						.attr('x2', xcoord)
						.attr('y2', ycoord)
						.attr('stroke', 'black')
						.attr('stroke-width', '2')
						.attr('marker-end', 'url(#arrowhead)');
					arrowList.push(arrow);
				});
				this.arrows[idx][jdx] = arrowList;
			});
		});
	}

	updateAgent() {
		this.grid.forEach(row => {
			row.forEach(tile => {
				if (!tile.legal || tile.goal) {
					return;
				}

				let xcoord = tile.x * this.d;
				let ycoord = tile.y * this.d;

				let qSum = 0;
				this.arrow_actions.forEach(a => {
					qSum += Math.pow(Math.E, tile.info[a]);
				});

				this.arrow_actions.forEach(a => {
					let lineSize = this.d / 2 * Math.pow(Math.E, tile.info[a]) / qSum;
					let dx = 0;
					let dy = 0;
					if (a < 2) {
						dx = Math.pow(-1, a + 1) * lineSize;
					} else {
						dy = Math.pow(-1, a) * lineSize;
					}

					if (lineSize > 3) {
						this.arrows[tile.x][tile.y][a]
							.attr('x1', xcoord + this.d / 2)
							.attr('y1', ycoord + this.d / 2)
							.attr('x2', xcoord + this.d / 2 + dx)
							.attr('y2', ycoord + this.d / 2 - dy)
							.attr('visibility', 'visible');
					} else {
						this.arrows[tile.x][tile.y][a]
							.attr('visibility', 'hidden');
					}
				});
			});
		});
		this.grid[this.pos_trace[this.time].x][this.pos_trace[this.time].y]
			.info[this.a_trace[this.time]] = this.model_trace[this.time];

		if (this.time % ((this.t_max + 1) / this.jumps) != 0) {
			return;
		}

		let index = this.time / ((this.t_max + 1) / this.jumps);
		if (this.time == 0) {
			this.grid.forEach(row => {
				row.forEach(tile => {
					tile.info = [0, 0, 0, 0];
				});
			});
		} else {
			for (let [key, value] of this.q_list[index].map) {
				let coord = {
					x: key.charAt(0),
					y: key.charAt(1),
				};
				let a = key.charAt(2);
				this.grid[coord.x][coord.y].info[a] = value;
			}
		}
	}
}

class BayesGridVis extends GridVisualization {
	constructor(env, trace, ui) {
		super(env, trace, ui);
		this.weights = [];
		this.color_normalization = this.N * this.N ;
		this.ig_trace = trace.ig;
	}

	updateAgent() {
		this.grid.forEach(row => {
			row.forEach(tile => {
				let rectangle = this.rectangles[tile.x][tile.y];
				let c = this.posteriorColor(tile, this.time);
				let col = null;
				if (c) {
					let r = Math.floor(c.r);
					let g = Math.floor(c.g);
					let b = Math.floor(c.b);
					col = `rgb(${r},${g},${b})`;
				} else {
					col = tile.color;
				}

				rectangle.attr('fill', col);
			});
		});
	}

	posteriorColor(tile, time) {
		let tc = tile.constructor;
		if (tc == Wall ||
			tc == SelfModificationTile) {
			return null;
		}

		if (tc == NoiseTile) {
			return {
				r: 255 * Math.random(),
				g: 255 * Math.random(),
				b: 255 * Math.random(),
			};
		}

		let trap = tile.constructor == Trap;

		// TODO visualize direct from agent mixture model!

		let p = this.model_trace[time][tile.y * this.N + tile.x];
		return {
			g: 255 - 200 * trap,
			r: 255 - p * this.color_normalization,
			b: 255 - p * this.color_normalization - 200 * trap,
		};
	}
}

BayesGridVis.exps = ['aixi', 'dispenser', 'mixture'];

class DirichletVis extends BayesGridVis {
	constructor(env, trace, ui) {
		super(env, trace, ui);
		this.model_trace = trace.params;
	}

	posteriorColor(tile, time) {
		let alphas = this.model_trace[time][tile.x][tile.y];
		let as = Util.sum(alphas);
		if (as == 0) {
			return { r: 255, g: 255, b: 255 };
		}

		return {
			r: 255 * (0.5 - alphas[2] / as),
			g: 255 * (1 - alphas[0] / as),
			b: 255 * (1 - alphas[1] / as),
		};
	}
}

DirichletVis.exps = ['dispenser', 'dirichlet'];

class ThompsonVis extends BayesGridVis {
	constructor(env, trace, ui) {
		super(env, trace, ui);
		this.rho_trace = trace.rhos;
	}

	updateAgent() {
		super.updateAgent();
		d3.select('#thompson_disp').remove();
		let rhoPos = this.rho_trace[this.time];
		GridVisualization.addCircle(
			this.svg, rhoPos.x, rhoPos.y, GridVisualization.colors.rho, 'thompson_disp', 1, this);
		d3.select('#agent').moveToFront();
	}
}

ThompsonVis.exps = ['dispenser', 'mixture', 'thompson'];

class MDLVis extends ThompsonVis { }

MDLVis.exps = ['dispenser', 'mixture', 'mdl'];

class BayesExpVis extends BayesGridVis {
	// TODO flag to show when agent is in explore mode or not
	// use trace.exploration_phases
}

BayesExpVis.exps = ['dispenser', 'mixture', 'bayesexp'];

class InqVis extends BayesGridVis {
	// TODO flag to show when agent is in explore mode or not
	// use trace.exploration_phases
}

Inq.exps = ['dispenser', 'mixture', 'inq'];

class MenteeVis extends BayesGridVis {
	// TODO flag to show when agent is in explore mode or not
	// use trace.exploration_phases
}

Mentee.exps = ['dispenser', 'mixture', 'mentee'];

class MentorVis extends BayesGridVis {
	// TODO flag to show when agent is in explore mode or not
	// use trace.exploration_phases
}

Mentor.exps = ['dispenser', 'mentor'];

class WireHeadVis extends BayesGridVis {
	updateAgent() {
		super.updateAgent();
		if (this.pos_trace[this.time].wireheaded) {
			for (let i = 0; i < this.N; i++) {
				for (let j = 0; j < this.N; j++) {
					let r = this.rectangles[i][j];
					let col = r.attr('fill');
					let rgb = col.replace(/[^\d,]/g, '').split(',');
					rgb[0] += 20;
					r.attr('fill', `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
				}
			}
		}
	}
}

WireHeadVis.exps = ['wirehead', 'mixture', 'dispenser'];

class HookedOnNoiseVis extends BayesGridVis { }

HookedOnNoiseVis.exps = ['dispenser', 'mixture', 'noise'];

class RewardCorruptionVis extends GridVisualization { }

RewardCorruptionVis.exps = ['reward_corruption'];
