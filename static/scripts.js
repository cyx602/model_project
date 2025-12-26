document.querySelectorAll('.predict-form').forEach(form => {
    form.addEventListener('submit', function(e) {
        let isValid = true;
        let errorMessage = "";
        const inputs = this.querySelectorAll('input');

        inputs.forEach(input => {
            const val = input.value.trim();
            const numVal = parseFloat(val);
            const min = parseFloat(input.getAttribute('min'));
            const max = parseFloat(input.getAttribute('max'));

            // 1. 检查是否为空或非数字
            if (val === "" || isNaN(numVal)) {
                input.classList.add('invalid');
                isValid = false;
                errorMessage = "Please enter valid numerical features!";
            }
            // 2. 检查是否超出范围 (新增)
            else if (numVal < min || numVal > max) {
                input.classList.add('invalid');
                isValid = false;
                errorMessage = `Value out of range! Expected: ${min} to ${max}`;
            }
            else {
                input.classList.remove('invalid');
            }
        });

        if (!isValid) {
            e.preventDefault(); // 拦截表单提交
            alert(errorMessage);
        }
    });
});