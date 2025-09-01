import React from "react";

const Spinner = () => {
  return (
    <div style={{
      display: "inline-block",
      width: "20px",
      height: "20px",
      border: "3px solid #ccc",
      borderTop: "3px solid #333",
      borderRadius: "50%",
      animation: "spin 1s linear infinite",
      marginRight: "8px",
    }} />
  );
};

export default Spinner;
