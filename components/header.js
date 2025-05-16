
let token = JSON.parse(localStorage.getItem("token"));
let name = JSON.parse(localStorage.getItem("userName"));


let change = "LOG IN"

if (token) {
  change = "Hi " + name;
}

function header() {
  return `
  
  <div class="hamburger">
    <span class="fas fa-bars"></span>
  </div>
  <div class="logo">
    <a href="./index.html"
      ><img
        height="140"
        width="280"
        src=""
        alt=""
    /></a>


    
    <span class="rightoptions">
    <a href="./search.html">
    <p id="search-input">SEARCH</p></a>
      <a href="login.html">
        <p id="userCheck">${change}</p>
      </a>
      <a href="help.html">
        <p>HELP</p>
      </a>
      <a href="shoppingBasket.html">
        <p>CART</p>
      </a>
    </span>
  </div>
}
