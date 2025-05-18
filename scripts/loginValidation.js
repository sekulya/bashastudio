let token = JSON.parse(localStorage.getItem("token"));

if (token) {
  let res = prompt(
    "You are already logged in.\nType 'yes' to logout, or press Cancel to stay logged in."
  );

  if (res === null) {
    // User pressed cancel
    window.location.href = "./index.html";
  } else {
    res = res.trim().toLowerCase();
    if (res === "yes") {
      localStorage.removeItem("token");
      alert("You have been logged out.");
    }
    window.location.href = "./index.html";
  }
}
