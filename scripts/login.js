let cart = JSON.parse(localStorage.getItem("cart")) || [];

function clickForPass() {
    document.querySelector("#inputEmail").classList.toggle("hidden");
    document.querySelector("#inputPasswd").classList.toggle("hidden");
}

function login(event) {
    event.preventDefault();
    
    let form = event.target;
    let formData = new FormData(form);
    let data = Object.fromEntries(formData);
    
    loginUser(data)
        .catch(error => {
            console.error("Login failed:", error);
            alert("Login failed. Please try again.");
        });
}

async function loginUser(data) {
    try {
        let res = await fetch("https://bashastudio.online/api/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        });

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        let response = await res.json();

        if (response.message) {
            alert(response.message);
            return;
        }

        alert("Login Successful 😊✌");

        localStorage.setItem("userName", JSON.stringify(response.user?.name));
        localStorage.setItem("token", JSON.stringify(response.token));

        window.location.href = cart.length ? "shipping.html" : "index.html";
        
    } catch (error) {
        console.error("Login error:", error);
        alert("Login failed. Please try again.");
        throw error;
    }
}
