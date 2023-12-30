var app = angular.module('rouletteApp', []);

app.controller('MainController', function() {
    this.batchInput = '';
    this.singleInput = '';
    this.streaks = {
        'Red Streak': 0,
        'Black Streak': 0,
        'Odd Streak': 0,
        'Even Streak': 0,
        'First Dozen Streak': 0,
        'Second Dozen Streak': 0,
        'Third Dozen Streak': 0,
        'First Column Streak': 0,
        'Second Column Streak': 0,
        'Third Column Streak': 0,
        'First Half Streak': 0,
        'Second Half Streak': 0
    };
    this.last15Numbers = [];
    this.redNumbers = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);

    this.submitBatch = function() {
        var numbers = this.batchInput.split(',').map(function(num) {
            return parseInt(num.trim());
        });
        numbers.forEach((num) => this.processNumber(num));
    };

    this.submitSingle = function() {
        this.processNumber(parseInt(this.singleInput));
    };

    this.processNumber = function(number) {
        // Update streaks
        this.updateStreaks(number);

        // Update last 15 numbers
        this.updateLast15(number);

        this.updateFilteredSortedStreaks();
    };

    this.updateStreaks = function(number) {
        if (number === 0 || number === -1) {
            this.increaseAllStreaks();
        } else {
            this.updateRedBlackStreaks(number);
            this.updateOddEvenStreaks(number);
            this.updateDozenStreaks(number);
            this.updateColumnStreaks(number);
            this.updateHalfStreaks(number);
        }
    };

    this.updateHalfStreaks = function(number) {
        if (number >= 1 && number <= 18) {
            this.streaks['First Half Streak'] = 0;
            this.streaks['Second Half Streak']++;
        } else if (number >= 19 && number <= 36) {
            this.streaks['First Half Streak']++;
            this.streaks['Second Half Streak'] = 0;
        } else {
            // If the number is 0, increase both half streaks
            this.streaks['First Half Streak']++;
            this.streaks['Second Half Streak']++;
        }
    };

    this.filteredSortedStreaks = [];

    this.updateFilteredSortedStreaks = function() {
        var filteredStreaks = [];
        for (var streak in this.streaks) {
            if (this.streaks[streak] > 0) {
                filteredStreaks.push({ name: streak, length: this.streaks[streak] });
            }
        }

        // Sorting the streaks in descending order
        filteredStreaks.sort(function(a, b) {
            return b.length - a.length;
        });

        this.filteredSortedStreaks = filteredStreaks;
    };

    this.updateRedBlackStreaks = function(number) {
        if (this.redNumbers.has(number)) {
            this.streaks['Red Streak'] = 0;
            this.streaks['Black Streak']++;
        } else {
            this.streaks['Red Streak']++;
            this.streaks['Black Streak'] = 0;
        }
    };

    this.updateOddEvenStreaks = function(number) {
        if (number % 2 === 0) {
            this.streaks['Even Streak'] = 0;
            this.streaks['Odd Streak']++;
        } else {
            this.streaks['Even Streak']++;
            this.streaks['Odd Streak'] = 0;
        }
    };

    this.increaseAllStreaks = function() {
        for (var streak in this.streaks) {
            this.streaks[streak]++;
        }
    };


    this.updateDozenStreaks = function(number) {
        if (number >= 1 && number <= 12) {
            this.streaks['First Dozen Streak'] = 0;
            this.streaks['Second Dozen Streak']++;
            this.streaks['Third Dozen Streak']++;
        } else if (number >= 13 && number <= 24) {
            this.streaks['First Dozen Streak']++;
            this.streaks['Second Dozen Streak'] = 0;
            this.streaks['Third Dozen Streak']++;
        } else if (number >= 25 && number <= 36) {
            this.streaks['First Dozen Streak']++;
            this.streaks['Second Dozen Streak']++;
            this.streaks['Third Dozen Streak'] = 0;
        } else {
            // If the number is 0, increase all dozen streaks
            this.streaks['First Dozen Streak']++;
            this.streaks['Second Dozen Streak']++;
            this.streaks['Third Dozen Streak']++;
        }
    };

    this.updateColumnStreaks = function(number) {
        if (number % 3 === 1 && number !== -1) {
            this.streaks['First Column Streak'] = 0;
            this.streaks['Second Column Streak']++;
            this.streaks['Third Column Streak']++;
        } else if (number % 3 === 2 && number !== -1) {
            this.streaks['First Column Streak']++;
            this.streaks['Second Column Streak'] = 0;
            this.streaks['Third Column Streak']++;
        } else if (number % 3 === 0 && number !== 0 && number !== -1) {
            this.streaks['First Column Streak']++;
            this.streaks['Second Column Streak']++;
            this.streaks['Third Column Streak'] = 0;
        } else if (number === 0 || number === -1) {
            // If the number is 0, increase all column streaks
            this.streaks['First Column Streak']++;
            this.streaks['Second Column Streak']++;
            this.streaks['Third Column Streak']++;
        }
    };

    this.updateLast15 = function(number) {
        this.last15Numbers.push({ number: number, color: this.getColor(number) });
        if (this.last15Numbers.length > 15) {
            this.last15Numbers.shift();
        }
    };

    this.getColor = function(number) {
        if (number === 0 || number === -1) return 'Green';
        return this.redNumbers.has(number) ? 'Red' : 'Black';
    };

    this.updateFilteredSortedStreaks();
    console.log(this.filteredSortedStreaks)
});
