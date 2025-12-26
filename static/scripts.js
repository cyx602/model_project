document.querySelectorAll('.predict-form').forEach(form => {
    form.addEventListener('submit', function(e) {
        let isValid = true;
        const inputs = this.querySelectorAll('input');

        inputs.forEach(input => {
            const val = input.value.trim();
            if (val === "" || isNaN(val)) {
                input.classList.add('invalid');
                isValid = false;
            } else {
                input.classList.remove('invalid');
            }
        });

        if (!isValid) {
            e.preventDefault();
            alert("Please enter valid numerical features!");
        }
    });
});