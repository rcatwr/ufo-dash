if (!window.dash_clientside) {
  window.dash_clientside = {};
}

window.dash_clientside.clientside = {
  sticky_func: function () {
    var header = document.querySelector("div.menu");

    // When the user scrolls the page, execute myFunction
    window.onscroll = function () {
      myFunction();
    };

    // Get the header
    var wrapper = document.querySelector("div.wrapper");

    // Get the offset position of the navbar
    var sticky = header.offsetTop;

    // Add the sticky class to the header when you reach its scroll position. Remove "sticky" when you leave the scroll position
    function myFunction() {
      if (window.pageYOffset > sticky) {
        header.classList.add("sticky");
        wrapper.classList.add('sticky-offset')
      } else {
        header.classList.remove("sticky");
        wrapper.classList.remove('sticky-offset')
      }
    }
  },
};
