class Mentor extends BayesAgent {
	constructor(options) {
		super(options);
		this.safe_spaces = [];
	}


	selectAction(e) {
		this.safe_spaces = [];
		for (var i = 0; i < this.numActions; i++) {
            if (e.traps[i] != Trap) 
            	{this.safe_spaces[this.safe_spaces.length] = i };
        }
    	if (this.safe_spaces == []) 
    		{ return 4; };
		return this.safe_spaces[Math.floor(Math.random() * this.safe_spaces.length)];
	}
}

Mentor.params = [
];
